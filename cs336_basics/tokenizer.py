import os
import regex as re
from typing import BinaryIO, Iterable
import time

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        """
            vocab: idx -> bytes
            merges: merged byte pairs during the training, preserving the original order
            special tokens: custom special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = { # For faster encoding
            value: i
            for i, value in enumerate(self.merges)
        }
        
        self.special_tokens: list[str] = ["<|endoftext|>"]
        if special_tokens: # add custom special_tokens
            deduplicated_custom_special_tokens = [sp for sp in special_tokens if sp not in self.special_tokens]

            initial_vocab_size = len(self.vocab.keys())
            for i in range(len(deduplicated_custom_special_tokens)):
                cur_special_token_in_bytes = deduplicated_custom_special_tokens[i].encode("utf-8")
                vocab[initial_vocab_size+i]=cur_special_token_in_bytes
            self.special_tokens += deduplicated_custom_special_tokens
            self.special_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True) # sort by length in descending order to avoid partial match during splitting
            
        # convert special tokens to bytes for fast encoding
        self.GPT2_regex = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_tokens_regex = re.compile(b"(" + "|".join(map(re.escape, self.special_tokens)).encode('utf-8') +b")") # escape |
        
        self.special_tokens_in_bytes: list[bytes] = [t.encode("utf-8") for t in self.special_tokens]
        # reserve a reverse vocab for fast decoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        vocab=None
        merges=None
        
        import pickle
        with open(vocab_filepath, 'rb') as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, 'rb') as mf:
            merges = pickle.load(mf)
        if vocab is None or merges is None:
            raise Exception("Empty vocab or merges")

        return cls(vocab, merges, special_tokens)

    def split_by_special_tokens_preserving(self, raw_text: str) -> list[bytes]:
        """
            Split raw text into documents by special tokens.
            The special tokens are removed from the documents.
            Each element of the returned list is either a document or special token
        """
        raw_text_in_bytes = raw_text.encode("utf-8")
        return re.split(self.special_tokens_regex, raw_text_in_bytes)
    
    def pre_tokenize(self, text_in_bytes: bytes) -> list[bytes]:
        """
            Perform the pre-tokenization on the given string.
            The input text should be guaranteed that they contain NO special token.
            It returns pre-tokenized tokens in bytes form.
        """
        return re.findall(self.GPT2_regex, text_in_bytes)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
            Current version always assume that the input iterable is a python file handler.
        """
        for text in iterable:
            yield from self.encode(text)

    def encode(self, text: str) -> list[int]:
        """
            In this function, we do not consider memory restriction.
            It works as the support function for `encode_iterable`, which is the memory-efficient encoding function for large files.
        """
        encoding_result: list[int] = []
        documents_and_special_tokens = self.split_by_special_tokens_preserving(text)
        for ele in documents_and_special_tokens: 
            # each ele is either a document without special token or a single special token
            if ele in self.special_tokens_in_bytes: 
                # if special token, directly look-up the reverse vocab and append to the result list
                encoding_result.append(self.reverse_vocab[ele])
                continue
            
            # otherwise, it's a piece of document
            pretokenized_doc = self.pre_tokenize(ele)
            for pre_token in pretokenized_doc:
                pre_token_in_single_bytes: list[bytes] = [bytes([b]) for b in pre_token]
                
                while len(pre_token_in_single_bytes) > 1:
                    bp_to_merge = min(zip(pre_token_in_single_bytes[:-1], pre_token_in_single_bytes[1:]), key=lambda x: self.merge_ranks.get(x, float('inf')))
                    if bp_to_merge not in self.merge_ranks: # no more byte pair to merge
                        break
                    
                    new_pre_token_in_single_bytes: list[bytes] = []
                    byte_idx = 0
                    while byte_idx < len(pre_token_in_single_bytes):
                        if byte_idx < len(pre_token_in_single_bytes) - 1 and (pre_token_in_single_bytes[byte_idx], pre_token_in_single_bytes[byte_idx + 1]) == bp_to_merge:
                            new_pre_token_in_single_bytes.append(bp_to_merge[0] + bp_to_merge[1])
                            byte_idx += 2
                        else: # bytes not merged
                            new_pre_token_in_single_bytes.append(pre_token_in_single_bytes[byte_idx])
                            byte_idx += 1
                    pre_token_in_single_bytes = new_pre_token_in_single_bytes
                
                encoding_result += [self.reverse_vocab[byte_piece] for byte_piece in pre_token_in_single_bytes]
        return encoding_result
                            
    def decode(self, ids: list[int]) -> str:
        input_in_bytes = b"".join([self.vocab[id] for id in ids])
        return input_in_bytes.decode("utf-8", errors='replace')
 
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_file_chunk(file_path, start, end):
    """Generator that yields lines from a file between byte offsets start and end."""
    with open(file_path, "rb") as file:
        file.seek(start)
        while file.tell() < end:
            line = file.readline()
            if not line:
                break
            yield line
            
def encode_chunk(file_path: str, start: int, end: int, tokenizer: Tokenizer) -> list[int]:
    """Encodes a chunk of a file defined by byte offsets start and end using the provided tokenizer."""
    token_ids = []
    for line in get_file_chunk(file_path, start, end):
        line_str = line.decode("utf-8", errors='replace')
        token_ids.extend(tokenizer.encode(line_str))
    return token_ids 
    
def tokenizer_experiments():
    SPECIAL_TOKEN = b"<|endoftext|>"
    # Initialize tokenizers
    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath='', # DEBUG
        merges_filepath='', # DEBUG
    )
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath='', # DEBUG
        merges_filepath='', # DEBUG
    )
    
    # experiment (a)
    print("Experiment (a):")
    # sample 10 documents from TinyStories and OpenWebText and check compression ratio
    with open("", "rb") as tsf,\
        open("", "rb") as owtf: # DEBUG
        # find <|endoftext|> from beginning to sample first 10 docs from each corpus
        ts_sample_docs: list[bytes] = []
        owt_sample_docs: list[bytes] = []
        st_count = 0
        for line in tsf:
            ts_sample_docs.append(line)
            if st_count + line.count(SPECIAL_TOKEN) >= 11:
                break
            st_count += line.count(SPECIAL_TOKEN)
        st_count = 0
        for line in owtf:
            owt_sample_docs.append(line)
            if st_count + line.count(SPECIAL_TOKEN) >= 11:
                break
            st_count += line.count(SPECIAL_TOKEN) 
    st_count = 0
    ts_sample_docs_in_bytes = b''
    owt_sample_docs_in_bytes = b''
    for piece in ts_tokenizer.split_by_special_tokens_preserving(b"".join(ts_sample_docs).decode("utf-8", errors='replace')):
        if piece == SPECIAL_TOKEN:
            st_count += 1
        if st_count >= 10:
            break
        ts_sample_docs_in_bytes += piece
    st_count = 0
    for piece in owt_tokenizer.split_by_special_tokens_preserving(b"".join(owt_sample_docs).decode("utf-8", errors='replace')):
        if piece == SPECIAL_TOKEN:
            st_count += 1
        if st_count >= 10:
            break
        owt_sample_docs_in_bytes += piece
        
    ts_original_byte_size = len(ts_sample_docs_in_bytes)
    owt_original_byte_size = len(owt_sample_docs_in_bytes)
    
    ts_sample_docs_str = ts_sample_docs_in_bytes.decode("utf-8", errors='replace')
    owt_sample_docs_str = owt_sample_docs_in_bytes.decode("utf-8", errors='replace')
    ts_encoded_token_list_size = len(ts_tokenizer.encode(ts_sample_docs_str))
    owt_encoded_token_list_size = len(owt_tokenizer.encode(owt_sample_docs_str))
    
    print(f"TinyStories: original byte size = {ts_original_byte_size}, encoded token list size = {ts_encoded_token_list_size}, compression ratio = {ts_original_byte_size/ts_encoded_token_list_size:.2f}")
    print(f"OpenWebText: original byte size = {owt_original_byte_size}, encoded token list size = {owt_encoded_token_list_size}, compression ratio = {owt_original_byte_size/owt_encoded_token_list_size:.2f}")
    
    # Experiment (b)
    print("Experiment (b):")
    # Try to tokenize sample docs from owt using tinystory tokenizer
    ts_encoded_token_list_size_for_owt_sample_docs = len(ts_tokenizer.encode(owt_sample_docs_str))
    print(f"Encoding OpenWebText sample docs using TinyStories tokenizer: encoded token list size = {ts_encoded_token_list_size_for_owt_sample_docs}, compression ratio = {owt_original_byte_size/ts_encoded_token_list_size_for_owt_sample_docs:.2f}")
    # Additionaly, let's do the reverse: tokenize sample docs from tinystory using owt tokenizer
    owt_encoded_token_list_size_for_ts_sample_docs = len(owt_tokenizer.encode(ts_sample_docs_str))
    print(f"Encoding TinyStories sample docs using OpenWebText tokenizer: encoded token list size = {owt_encoded_token_list_size_for_ts_sample_docs}, compression ratio = {ts_original_byte_size/owt_encoded_token_list_size_for_ts_sample_docs:.2f}")
    
    # Experiment (c)
    print("Experiment (c):")
    # Estimate the encoding throughput of the tokenizer
    import time
    NUM_RUNS = 10
    start_time = time.time()
    for _ in range(NUM_RUNS):
        _ = ts_tokenizer.encode(ts_sample_docs_str)
    end_time = time.time()
    avg_time_per_run = (end_time - start_time) / NUM_RUNS
    ts_throughput = len(ts_sample_docs_in_bytes) / avg_time_per_run # B/s
    print(f"TinyStories tokenizer encoding throughput: {ts_throughput:.2f} B/s")
    
    start_time = time.time()
    for _ in range(NUM_RUNS):
        _ = owt_tokenizer.encode(owt_sample_docs_str)
    end_time = time.time()
    avg_time_per_run = (end_time - start_time) / NUM_RUNS
    owt_throughput = len(owt_sample_docs_in_bytes) / avg_time_per_run # B/s
    print(f"OpenWebText tokenizer encoding throughput: {owt_throughput:.2f} B/s")
    
    # Cross-Tokenizer encoding throughput check
    start_time = time.time()
    for _ in range(NUM_RUNS):
        _ = ts_tokenizer.encode(owt_sample_docs_str)
    end_time = time.time()
    avg_time_per_run = (end_time - start_time) / NUM_RUNS
    ts_cross_throughput = len(owt_sample_docs_in_bytes) / avg_time_per_run # B/s
    print(f"TinyStories tokenizer encoding OpenWebText sample docs throughput: {ts_cross_throughput:.2f} B/s")
    
    start_time = time.time()
    for _ in range(NUM_RUNS):
        _ = owt_tokenizer.encode(ts_sample_docs_str)
    end_time = time.time()
    avg_time_per_run = (end_time - start_time) / NUM_RUNS
    owt_cross_throughput = len(ts_sample_docs_in_bytes) / avg_time_per_run # B/s
    print(f"OpenWebText tokenizer encoding TinyStories sample docs throughput: {owt_cross_throughput:.2f} B/s")
    
    # Experiment (d)
    print("Experiment (d):")
    NUM_PROCESS = 10 # DEBUG
    from multiprocessing import Pool
    import numpy as np
    
    for tokenizer, file_path in [
        (ts_tokenizer, ""),
        (ts_tokenizer, ""),
        (owt_tokenizer, ""),
        (owt_tokenizer, ""),
    ]:
        start_time=time.time()
        print(f'tokenizing for {file_path}...')
        with open(file_path, "rb") as file:
            chunk_boundaries = find_chunk_boundaries(file, NUM_PROCESS, split_special_token=SPECIAL_TOKEN)
        print(f"Chunk boundaries are found for {file_path}")
        
        # multiprocessing encoding
        print("Start multiprocessing encoding...")
        encoding_result = []
        with Pool(processes=NUM_PROCESS) as pool:
            for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
                encoding_result.append(pool.apply_async(encode_chunk, args=(file_path, start, end, tokenizer)))
            pool.close()
            pool.join()
           
        # Serialize the token IDs as a numpy array of datatype uint16.
        final_np_array_to_save = np.array([], dtype=np.uint16)
        for result in encoding_result:
            result_list = result.get()
            final_np_array_to_save = np.concatenate((final_np_array_to_save, np.array(result_list, dtype=np.uint16)))
        np.save(file_path.replace(".txt", "_encoded.npy"), final_np_array_to_save)
        print("Done! Time taken: {:.2f} seconds".format(time.time() - start_time))
            

if __name__ == "__main__":
    # tokenizer = Tokenizer.from_files(
    #     vocab_filepath='', # DEBUG
    #     merges_filepath='', # DEBUG
    #     special_tokens=["<|endoftext|><|endoftext|>"]
    # )
    # sample_text = '' # DEBUG
    # tokenizer.encode(sample_text)
    
    tokenizer_experiments()
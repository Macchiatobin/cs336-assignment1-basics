import os
import regex as re
import multiprocessing as mp
from typing import BinaryIO

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


def pre_tokenize(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[bytes, int]:
    token_counts = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start)
        
        # split and remove special tokens
        SPECIAL_TOKENS = "|".join(map(re.escape, special_tokens)).encode('utf-8') # escape |
        # pre-tokenization
        GPT2_REGEX = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # split by special tokens to obtain documents
        documents = re.split(SPECIAL_TOKENS, chunk_data)
        for doc in documents:
            tokens = re.finditer(GPT2_REGEX, doc)
            for token in tokens:
                token_bytes = token.group()
                if token_bytes in token_counts:
                    token_counts[token_bytes] += 1
                else:
                    token_counts[token_bytes] = 1   
    return token_counts

def merge_bp(
    vocab: dict, # bytes: idx
    reverse_vocab: dict, # idx: bytes
    merges: list[tuple[bytes, bytes]], # list of (bytes, bytes), merge history
    bp2token: dict, # byte pair: list of tokens containing the pair
    bp2occur: dict, # byte pair: occurrence count
    token2count: dict, # pre-token: count, remains unchanged
    token2sep: dict, # token: [vocab_idx]
):
    """perform single turn token merge"""
    bp_to_merge = max(bp2occur, key=lambda x: (bp2occur[x], x)) # pick largest occur. count. if tie, pick lexicographically larger bp
    merges.append(bp_to_merge) # new merge history
    # new token in vocab / reversed_vocab
    merged_bp=bp_to_merge[0]+bp_to_merge[1]
    new_bp_vocab_idx = len(vocab.keys())
    vocab[merged_bp]=new_bp_vocab_idx
    reverse_vocab[new_bp_vocab_idx]=merged_bp
    
    # original bp's vocab idx:
    merged_bp_original_vocab_idx_pair = (vocab[bp_to_merge[0]], vocab[bp_to_merge[1]])
    bp_occur_count = bp2occur.pop(bp_to_merge)
    tokens_with_bp_to_merge = bp2token.pop(bp_to_merge)
    
    for token in tokens_with_bp_to_merge:
        token_sep = token2sep[token]
        original_token_sep = token_sep
        
        # merge bps and get new token_sep
        if_exhausted = False
        while not if_exhausted: # whether we enumerated til the end of the current pre-token
            break_in_middle = False
            for in_token_idx, (a, b) in enumerate(zip(token_sep[:-1], token_sep[1:])): # collect pos of bp_to_merge in current pre-token
                if (a, b) == merged_bp_original_vocab_idx_pair:
                    # update new_token_sep as it merged
                    new_token_sep = [i for i in token_sep[:in_token_idx]]
                    new_token_sep.append(new_bp_vocab_idx)
                    new_token_sep_post = [i for i in token_sep[in_token_idx + 2:]]
                    new_token_sep += new_token_sep_post
                    token_sep = new_token_sep
                    break_in_middle = True
                    break # merge current bp and break, start new round of enumeration
            if not break_in_middle: # current pre-token exhausted
                if_exhausted = True
        # now token_sep is new_token_sep, in which the new bp is merged, update it here
        token2sep[token] = token_sep
        
        # DEBUG
        # print(f'current token: {token}')
        # print(f'Token Sep before merge: {original_token_sep}')
        # print(f'Token Sep after merge: {token_sep}')
        
        
        # check preceding / subsequent byte of merged bytes and adjust the bp2occur if any
        for i, b in enumerate(token_sep):
            if b == new_bp_vocab_idx:
                if i - 1 >= 0: # preceding
                    first_byte = reverse_vocab[token_sep[i - 1]]
                    second_byte = bp_to_merge[0]
                    cur_decreasing_byte_pair = (first_byte, second_byte) if first_byte != merged_bp else (bp_to_merge[1], second_byte)
                    # we decrease occur count of byte-pair before merge
                    if not bp_to_merge == cur_decreasing_byte_pair: 
                        # if same byte pair, already not in bp2occur (e.g.: AAA)
                        bp2occur[cur_decreasing_byte_pair] -= token2count[token]
                        
                    cur_increasing_byte_pair = (first_byte, merged_bp) # increase/set current pre-token's count
                    if cur_increasing_byte_pair in bp2occur.keys():
                        bp2occur[cur_increasing_byte_pair] += token2count[token]
                    else:
                        bp2occur[cur_increasing_byte_pair] = token2count[token]
                    if cur_increasing_byte_pair in bp2token.keys():
                        if token not in bp2token[cur_increasing_byte_pair]:
                            bp2token[cur_increasing_byte_pair].append(token)
                    else:
                        bp2token[cur_increasing_byte_pair] = [token]
                if i + 1 < len(token_sep): # subsequent
                    first_byte = bp_to_merge[1]
                    second_byte = reverse_vocab[token_sep[i + 1]]
                    cur_decreasing_byte_pair = (first_byte, second_byte) if second_byte != merged_bp else (first_byte, bp_to_merge[0])
                    # we decrease occur count of byte-pair before merge
                    if not bp_to_merge == cur_decreasing_byte_pair:
                        # same as counterpart of preceding byte checking
                        bp2occur[cur_decreasing_byte_pair] -= token2count[token]
                        
                    cur_increasing_byte_pair = (merged_bp, second_byte)
                    if second_byte == merged_bp:
                        continue # (merged_bp, merged_bp)'s count/apperance already increased in the previous pass (increasing subsequent)
                    
                    if cur_increasing_byte_pair in bp2occur.keys():
                        bp2occur[cur_increasing_byte_pair] += token2count[token]
                    else:
                        bp2occur[cur_increasing_byte_pair] = token2count[token]
                    if cur_increasing_byte_pair in bp2token.keys():
                        if token not in bp2token[cur_increasing_byte_pair]:
                            bp2token[cur_increasing_byte_pair].append(token)
                    else:
                        bp2token[cur_increasing_byte_pair] = [token]
                        
    
        # remove old values from bp2token
        new_bps_in_new_token_sep = [(a,b) for (a,b) in zip(token_sep[:-1], token_sep[1:])]
        for (a, b) in zip(original_token_sep[:-1], original_token_sep[1:]):
            if (a,b) != merged_bp_original_vocab_idx_pair and ((a, b) not in new_bps_in_new_token_sep): # bp_to_merge already popped
                if token in bp2token[(reverse_vocab[a],reverse_vocab[b])]:
                    bp2token[(reverse_vocab[a],reverse_vocab[b])].remove(token)
    # END-of-current-token
                

def bpe_tokenizer_training(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {} # bytes to idx
    merges = [] # merge records
    # initialize vocab
    for i in range(256):
        vocab[bytes([i])] = i
    for i in range(len(special_tokens)):
        vocab[special_tokens[i].encode("utf-8")] = 256+i
    reverse_vocab = {i:b for b, i in vocab.items()}
    
    # chunk texts into independent chunks
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # multiprocessing to process chunks
    pre_tokenization_results_per_process = []
    with mp.Pool(processes=num_processes) as pool:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            pre_tokenization_results_per_process.append(pool.apply_async(pre_tokenize, args=(input_path, start, end, special_tokens)))
        pool.close()
        pool.join()
        
    # aggregate
    token2count = {} # token occurrence count
    token2sep = {} # each token's current separation
    for result in pre_tokenization_results_per_process:
        cur_token_count: dict[bytes, int] = result.get()
        for token_bytes, count in cur_token_count.items():
            if token_bytes in token2count.keys():
                token2count[token_bytes] += count
            else:
                token2count[token_bytes] = count
    for token in token2count.keys():
        token2sep[token] = [b for b in token]
    # print("Aggregated token2count sample:") # DEBUG
    # print(list(token2count.items())[:5])
    # print("...")
    # print(list(token2count.items())[-5:])
    # print(len(token2count.keys()))
    # print("token2sep sample:")
    # print(list(token2sep.items())[:5])
    # print("...")
    # print(list(token2sep.items())[-5:])
    
    
    # cache
    bp2token = {}  # byte pairs to pre-token value
    bp2occur = {} # byte pairs to occurrence count
    for token in token2count.keys(): # multiple bp occurrence within same pre-token is considered
        for a, b in zip(token[:-1], token[1:]):
            cur_pair = (bytes([a]), bytes([b]))
            if cur_pair not in bp2token:
                bp2token[cur_pair] = [token]
                bp2occur[cur_pair] = token2count[token]
            else:
                if token not in bp2token[cur_pair]: # de-duplication
                    bp2token[cur_pair].append(token)
                bp2occur[cur_pair] += token2count[token]
    # print("Initial byte-pair2token sample:") # DEBUG
    # print(list(bp2token.items())[:5])
    # print("...")
    # print(list(bp2token.items())[-5:])
    # print("Initial byte-pair2occur sample:")
    # print(list(bp2occur.items())[:5])
    # print("...")
    # print(list(bp2occur.items())[-5:])
    # print("Initial token2count sample:")
    # print(list(token2count.items())[:5])
    # print("...")
    # print(list(token2count.items())[-5:])
    
    # do merge
    while len(vocab) < vocab_size:
        merge_bp(
            vocab,
            reverse_vocab,
            merges,
            bp2token,
            bp2occur,
            token2count,
            token2sep,
        )
    return reverse_vocab, merges

if __name__ == "__main__":
    # BPE training on TinyStories part
    # input_path = "/home/canbin/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000
    # BPE training on OpenWebText part
    input_path = "/home/canbin/cs336-assignment1-basics/data/owt_valid.txt"
    vocab_size = 32000
    
    special_tokens = ["<|endoftext|>"]
    
    vocab, merges = bpe_tokenizer_training(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    print("Vocabulary:")
    print(list(vocab.items())[:5])
    print("...")
    print(list(vocab.items())[-5:])
    print("Merges:")
    print(merges[:5])
    print("...")
    print(merges[-5:])
    
    # serialize vocab and merges to disk
    import pickle
    with open("bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
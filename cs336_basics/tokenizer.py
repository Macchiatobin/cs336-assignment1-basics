import regex as re
from typing import Iterable

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        """
            vocab: idx -> bytes
            merges: merged byte pairs during the training, preserving the original order
            special tokens: custom special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens: list[str] = ["<|endoftext|>"]
        if special_tokens: # add custom special_tokens
            deduplicated_custom_special_tokens = [sp for sp in special_tokens if sp not in self.special_tokens]

            initial_vocab_size = len(self.vocab.keys())
            for i in range(len(special_tokens)):
                cur_special_token_in_bytes = special_tokens[i].encode("utf-8")
                vocab[initial_vocab_size+i]=cur_special_token_in_bytes
            self.special_tokens += special_tokens
            
        # convert special tokens to bytes for fast encoding
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
        SPECIAL_TOKENS = "|".join(map(re.escape, self.special_tokens)).encode('utf-8') # escape |
        SPECIAL_TOKENS_TO_SPLIT = b"(" + SPECIAL_TOKENS + b")" # preserve special tokens too, as they should be encoded as well
        raw_text_in_bytes = raw_text.encode("utf-8")
        documents=re.split(SPECIAL_TOKENS_TO_SPLIT, raw_text_in_bytes)
        return documents
    
    def pre_tokenize(self, raw_text: str) -> list[bytes]:
        """
            Perform the pre-tokenization on the given string.
            The input text should be guaranteed that they contain NO special token.
            It returns pre-tokenized tokens in bytes.
        """
        GPT2_REGEX = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        raw_text_in_bytes = raw_text.encode("utf-8")
        tokens = re.finditer(GPT2_REGEX, raw_text_in_bytes)
        pretokenized_bytes = [t.group() for t in tokens]
        return pretokenized_bytes

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
            Current version always assume that the input iterable is a python file handler.
        """
        for text in iterable:
            yield self.encode(text)

    def encode(self, text: str) -> list[int]:
        """
            In this function, we do not consider memory restriction.
            It works as the support function for `encode_iterable`, which is the memory-efficient encoding function for large files.
        """
        encoding_result = []
        documents_and_special_tokens = self.split_by_special_tokens_preserving(text)
        for ele in documents_and_special_tokens: 
            # each ele is either a document without special token or a single special token
            if ele in self.special_tokens_in_bytes: 
                # if special token, directly look-up the reverse vocab and append to the result list
                encoding_result += self.reverse_vocab[ele]
                continue
            
            # otherwise, it's a piece of document
            pretokenized_doc = self.pre_tokenize(ele)
            for pre_token in pretokenized_doc:
                pre_token_in_single_bytes = [bytes(b) for b in pre_token]
                current_byte_pairs = [(a,b) for (a,b) in zip(pre_token_in_single_bytes[:-1], pre_token_in_single_bytes[1:])]
                while len(pre_token_in_single_bytes) > 1:
                    for merge in self.merges:
                        if merge in current_byte_pairs:
                            pass
                            # TODO: do all merge in current_byte_paris and update
                            

    
    def decode(self, ids: list[int]) -> str:
        input_in_bytes = bytes()
        for id in ids:
            input_in_bytes += self.vocab[id]
        return input_in_bytes.decode("utf-8", errors='replace')
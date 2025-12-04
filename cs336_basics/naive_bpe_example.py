# Simple implementation of bpe encoding training process

# helper for debugging
def show_freq_count(freq_count, reverse_vocab):
    print("showing current freq_count:")
    for token, count in freq_count.items():
        token_tuple = tuple([reverse_vocab[t] for t in token])
        print(f'Token: {token_tuple}, Count: {count}')

# pre-tokenization for corpus to train on
CORPUS = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
PRE_TOKENIZATION = CORPUS.split(" ")
FREQ_COUNT = {} # FREQ
for token in PRE_TOKENIZATION:
    if tuple(bytes(token, encoding='utf-8')) in FREQ_COUNT:
        FREQ_COUNT[tuple(bytes(token, encoding='utf-8'))] += 1
    else:
        FREQ_COUNT[tuple(bytes(token, encoding='utf-8'))] = 1 # tuple of ints (vocab idx)

# prepare vocab 
VOCAB = {} # byte -> idx
REVERSE_VOCAB = {} # idx -> byte
for idx in range(256):
    VOCAB[bytes([idx])] = idx
    REVERSE_VOCAB[idx] = bytes([idx])
VOCAB[b"<|endoftext|>"] = 256
REVERSE_VOCAB[256] = b"<|endoftext|>"
MAX_VOCAB_KEY = b"<|endoftext|>"
MAX_VOCAB_VALUE = 256
print('Initial vocab:')
print(list(VOCAB.items())[:5], '...', list(VOCAB.items())[-5:])
print('Initial reverse_vocab:')
print(list(REVERSE_VOCAB.items())[:5], '...', list(REVERSE_VOCAB.items())[-5:])
print("max vocab key:", MAX_VOCAB_KEY)
print("max vocab value:", MAX_VOCAB_VALUE)
print('Initial frequency count:')
show_freq_count(FREQ_COUNT, REVERSE_VOCAB)

# start traininng bpe merges (training)
print('Starting BPE merges...')
N_MERGES = 6
for merge_count in range(N_MERGES):
    print(f'{merge_count+1} / {N_MERGES} merge starts!')
    current_pairs = {}
    for token in FREQ_COUNT:
        for pair in range(len(token)-1):
            current_pair = (token[pair], token[pair+1]) # get idx pair
            if current_pair in current_pairs.keys():
                current_pairs[current_pair] += FREQ_COUNT[token]
            else:
                current_pairs[current_pair] = FREQ_COUNT[token]
    print(current_pairs)            
    max_key = max(current_pairs, key=lambda x: (current_pairs[x], x)) # idx pair, pick largest value then lexicographically greater if there's tie
    max_value = current_pairs[max_key] # frequency
    MAX_VOCAB_KEY = REVERSE_VOCAB[max_key[0]] + REVERSE_VOCAB[max_key[1]] # new byte pair
    MAX_VOCAB_VALUE += 1
    print(f'Most frequent pair: {MAX_VOCAB_KEY} (key={max_key}) with count {max_value}') # for clarity
    VOCAB[MAX_VOCAB_KEY] = MAX_VOCAB_VALUE # new pair to vocab
    REVERSE_VOCAB[MAX_VOCAB_VALUE] = MAX_VOCAB_KEY
    print(f'New vocab entry: {MAX_VOCAB_KEY} with index {MAX_VOCAB_VALUE}')
    
    NEW_FREQ_COUNT = {}
    for token, value in FREQ_COUNT.items():
        pair_hit_idx = []
        for idx in range(len(token) - 1):
            current_pair = (token[idx], token[idx + 1]) # current pair of idx
            if current_pair == max_key: # hit
                pair_hit_idx.append(idx)
        
        new_token = []
        # if any pair hit, reconstruct current token
        if pair_hit_idx:
            skipping=False
            for i in range(len(token)):
                if skipping:
                    skipping=False # skip subsequent byte when pair idx hit
                    continue
                if i in pair_hit_idx:
                    new_token.append(MAX_VOCAB_VALUE) 
                    skipping=True # skipping next byte
                else:
                    new_token.append(token[i])
        else:
            new_token=token
        new_token = tuple(new_token) # dict key should be tuple
        
        NEW_FREQ_COUNT[new_token] = value # count remains the same
    FREQ_COUNT = NEW_FREQ_COUNT # overwrite the frequency count
    print(f'Frequency count after current merge:')
    show_freq_count(FREQ_COUNT, REVERSE_VOCAB)
    
print('BPE merges completed!')
print('Final vocab:')
print(list(VOCAB.items())[:10], '...', list(VOCAB.items())[-10:])
print('Final reverse_vocab:')
print(list(REVERSE_VOCAB.items())[:10], '...', list(REVERSE_VOCAB.items())[-10:])
print('Final frequency count:')
show_freq_count(FREQ_COUNT, REVERSE_VOCAB)
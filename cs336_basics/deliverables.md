# Deliverables

## BPE Training on TinyStories

a. It took about 1.5 min (1 minute 28 seconds) and 270MB memory as peak value, shown from the result of scalene profile.
From the deserialized vocab, I could find out that the longest token was 'accomplishment' and it does make sense.

b. Pre-tokenization phase took the most of the time. To be more precise, the iterating and counting token's occurrence count during the pre-tokenization phase took the most.

## BPE Training on OpenWebText

a. The longest token I obtained from OpenWebText is '----------------------------------------------------------------'. It makes less sense than the one I got from `TinyStories`. But considering the randomness and noisiness of the web text, it is, still, a plausible outcome.

b. In general, I could observe that some words that could be considered as 'informal' from the vocabulary derived from OpenWebText. This reflects the importance of the original training corpus and its influence on the tokenizer.

## Experiments with tokenizers

a. The TinyStory tokenizer's compression ratio is $4.04$ and OpenWebText tokenizer's is $4.52$, which means the latter performes better.

b. When tokenizing sample documents from OpenWebText using TinyStory tokenizer, the compression ratio further drops from $4.04$ to $3.41$. Additionaly, I did the reverse, i.e. tokenize TinyStory samples using OpenWebText tokenizer and found out that the compression ratio also dropped from $4.52$ to $3.89$. This shows that the tokenizers trained using different corpus performe better, or to say the best, on the their original corpus.

c. In general, the throughput was $0.30\text{~}0.36\text{MB/s}$. It means that the estimated time cost for tokenizing 825$GB$-sized Pile dataset will be over $700$ hours or so.

d. The range of `uint16` is $0\text{~}65,535$. And the size of our tokenizers' vocabs don't exceed 4k. So `uint16` is the smallest `int` type that fits our vocab sizes, which makes it the most desirable datatype, when considering the storage efficiency.
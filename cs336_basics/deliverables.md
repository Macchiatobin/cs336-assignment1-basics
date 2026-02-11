# Deliverables

## BPE Training on TinyStories

a. It took about 1.5 min (1 minute 28 seconds) and 270MB memory as peak value, shown from the result of scalene profile.
From the deserialized vocab, I could find out that the longest token was 'accomplishment' and it does make sense.

b. Pre-tokenization phase took the most of the time. To be more precise, the iterating and counting token's occurrence count during the pre-tokenization phase took the most.

## BPE Training on OpenWebText

a. The longest token I obtained from OpenWebText is '----------------------------------------------------------------'. It makes less sense than the one I got from `TinyStories`. But considering the randomness and noisiness of the web text, it is, still, a plausible outcome.

b. In general, I could observe that some words that could be considered as 'informal' from the vocabulary derived from OpenWebText. This reflects the importance of the original training corpus and its influence on the tokenizer.
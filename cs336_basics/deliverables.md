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

## Transformer LM resource accounting

### Parameter & FLOP accounting

Before everything, all parameters (learnable & unlearnable but needs loading) and corresponding FLOPs (needed for single sequence's inference only) within each component of transformers are listed here. The sublists listed below each component stands for that of single layer.

> Assuming that model's configuration, such as `vocab_size`, `d_model`, `d_head`, `seq_len`, `d_ff` are given.

1. Token Embeddings
    - Learnable Parameter
        - Embedding matrix: `(vocab_size, d_model)`
        - Total: `vocab_size * d_model`
    - Unlearnable Parameter: None
    - FLOPs: It's pure table look-up by index slicing, no FLOP is needed for this layer
2. RMSNorm
    - Learnable Parameter
        - Gain parameter: `(d_model,)`
        - Total: `d_model`
    - Unlearnable Parameter: None, empirically take `eps=1e-5`
    - FLOPs: `4 * seq_len * d_model`
        - `RMS` calculation: `2 * seq_len * d_model`
        - Applying normalization: `2 * seq_len * d_model`
3. Causal Multi-Head Self-Attention w/ RoPE
    - Learnable Parameter
        - Query projection weight: `(d_model, d_model)`
        - Key projection weight: `(d_model, d_model)`
        - Value projection weight: `(d_model, d_model)`
        - Output projection weight: `(d_model, d_model)`
    - Unlearnable Parameter
        - RoPE cos & sin buffer: `(2, max_seq_len, d_head)`
    - FLOPs: `8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model + 6 * seq_len * d_model + 5 * seq_len * seq_len * n_head`
        - Query,Key,Value projection: `6 * seq_len * d_model * d_model`
        - RoPE per head: `6 * seq_len * d_head`, in total: `6 * seq_len * d_model`
        - Scaled dot product attention: 
            - `QK^T` per head: `2 * seq_len * seq_len * d_head`, in total: `2 * seq_len * seq_len * d_model`
            - `sqrt(d_head)` scaling per head: `seq_len * seq_len`, in total: `seq_len * seq_len * n_head`
            - softmax per head: `4 * seq_len * seq_len`, in total: `4 * seq_len * seq_len * n_head`
            - attention output per head: `2 * seq_len * seq_len * d_head`, in total: `2 * seq_len * seq_len * d_model`
        - Output projection: `2 * seq_len * d_model * d_model`
4. Position-Wise Feed-Forward Network
    - Learnable Parameter
        - `W1`: `(d_model, d_ff)`
        - `W2`: `(d_ff, d_model)`
        - `W3`: `(d_model, d_ff)`
        - Total: `3 * d_model * d_ff`
    - Unlearnable Parameter: None
    - FLOPs: `6 * seq_len * d_ff * (d_model + 1)`
        - W1 projection: `2 * seq_len * d_model * d_ff`
        - Swish activation: `5 * seq_len * d_ff`
        - W3 projection: `2 * seq_len * d_model * d_ff`
        - Gate operation: `seq_len * d_ff`
        - Output projection: `2 * seq_len * d_ff * d_model`
5. LM Head (output embedding)
    - Learnable Parameter
        - Output projection layer: `(d_model, vocab_size)`
        - Total: `d_model * vocab_size`
    - Unlearnable Parameter: None
    - FLOPs: `2 * seq_len * d_model * vocab_size`
        - Output projection: `2 * seq_len * d_model * vocab_size`

### Problems

a. With given GPT-2 XL configuration, each transformer block will need: 2 RMSNorms (`2 * d_model`), Attention layer (`4 * d_model * d_model`), FFN (`3 * d_model * d_ff`), which means that each block's parameter number will be `40,963,200` and `1,966,233,600` (`2 * max_seq_len * d_head = 131,072` for rope omitted) for all layers. 
Besides transformer blocks, there are token embedding layer (`vocab_size * d_model = 80,411,200`), final norm (`d_model = 1600`), LM head layer (`vocab_size * d_model = 80,411,200`). 
In total, there will be `2,127,188,672` learnable parameters. When represented using single-precision floating point (`float32`), about `8GB` memory will be needed to just load the model.

b. Assuming that input sequence's length is `context_length`, each transformer block's forward-pass FLOP count includes: 2 RMSNorms ($\frac{x}{RMS(x)}*\gamma$, `8 * seq_len * d_model = 13,107,200`, `0.01%`), Attention layer w/ RoPE ($(softmax(\frac{Q_{rope}K_{rope}^T}{\sqrt{d_{model}}})V)W_O$, `8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model + 6 * seq_len * d_model + 5 * seq_len * seq_len * n_head = 27,823,308,800`, `30.6%`), FFN ($(SiLU(W_1x)\bigodot W_2x)W_3$, `6 * seq_len * d_ff * (d_model + 1) = 62,953,881,600`, `69.3%`), which in total will be `90,790,297,600` and `4,357,934,284,800` for all layers. 
Besides transformer blocks, there are final norm (`4 * seq_len * d_model = 6,553,600`, `0.0002%`), LM head layer (`2 * vocab_size * d_model = 160,822,400`, `0.004%`). 
In summary, `4,358,101,660,800` FLOPs are needed for the given sequence's forward pass.

c. Based on the analysis, FFN parts require the most FLOPs.

d. During the analysis below, assume that `context_length`, `d_ff` remain unchanged. 

GPT-2 small (12 layers, 768 `d_model`, 12 heads): each transformer block's forward-pass FLOP count includes: 2 RMSNorms ($\frac{x}{RMS(x)}*\gamma$, `8 * seq_len * d_model = 6,291,456`, `0.02%`), Attention layer w/ RoPE ($(softmax(\frac{Q_{rope}K_{rope}^T}{\sqrt{d_{model}}})V)W_O$, `8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model + 6 * seq_len * d_model + 5 * seq_len * seq_len * n_head = 8,120,696,832`, `21.2%`), FFN ($(SiLU(W_1x)\bigodot W_2x)W_3$, `6 * seq_len * d_ff * (d_model + 1) = 30,238,310,400`, `78.8%`), which in total will be `38,365,298,688` and `460,383,584,256` for all layers. Besides transformer blocks, there are final norm (`4 * seq_len * d_model = 3,145,728`, `0.0007%`), LM head layer (`2 * vocab_size * d_model = 77,194,752`, `0.02%`). So it will be `460,463,924,736`.

GPT-2 medium (24 layers, 1024 `d_model`, 16 heads): each transformer block's forward-pass FLOP count includes: 2 RMSNorms ($\frac{x}{RMS(x)}*\gamma$, `8 * seq_len * d_model = 8,388,608`, `0.02%`), Attention layer w/ RoPE ($(softmax(\frac{Q_{rope}K_{rope}^T}{\sqrt{d_{model}}})V)W_O$, `8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model + 6 * seq_len * d_model + 5 * seq_len * seq_len * n_head = 12,975,079,424`, `24.3%`), FFN ($(SiLU(W_1x)\bigodot W_2x)W_3$, `6 * seq_len * d_ff * (d_model + 1) = 40,304,640,000`, `75.6%`), which in total will be `53,288,108,032` and `1,278,914,592,768` for all layers. Besides transformer blocks, there are final norm (`4 * seq_len * d_model = 4,194,304`, `0.0003%`), LM head layer (`2 * vocab_size * d_model = 102,926,336`, `0.008%`). So it will be `1,279,021,713,408`.

GPT-2 large (36 layers, 1280 `d_model`, 20 heads): each transformer block's forward-pass FLOP count includes: 2 RMSNorms ($\frac{x}{RMS(x)}*\gamma$, `8 * seq_len * d_model = 10,485,760`, `0.02%`), Attention layer w/ RoPE ($(softmax(\frac{Q_{rope}K_{rope}^T}{\sqrt{d_{model}}})V)W_O$, `8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model + 6 * seq_len * d_model + 5 * seq_len * seq_len * n_head = 18,903,203,840`, `27.3%`), FFN ($(SiLU(W_1x)\bigodot W_2x)W_3$, `6 * seq_len * d_ff * (d_model + 1) = 50,370,969,600`, `72.7%`), which in total will be `69,284,659,200` and `2,494,247,731,200` for all layers. Besides transformer blocks, there are final norm (`4 * seq_len * d_model = 5,242,880`, `0.0002%`), LM head layer (`2 * vocab_size * d_model = 128,657,920`). So it will be `2,494,381,632,000`, `0.005%`. 

In sum, Parts of LM proportionally taking more as the model size increases: Attention layer, parts taking less: all others.

e. Taking GPT2-XL from (a) with increased context length (`16,384`), each transformer block's forward-pass FLOP count includes: 2 RMSNorms ($\frac{x}{RMS(x)}*\gamma$, `8 * seq_len * d_model = 209,715,200`, `0.007%`), Attention layer w/ RoPE ($(softmax(\frac{Q_{rope}K_{rope}^T}{\sqrt{d_{model}}})V)W_O$, `8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model + 6 * seq_len * d_model + 5 * seq_len * seq_len * n_head = 2,085,900,779,520`, `67.4%`), FFN ($(SiLU(W_1x)\bigodot W_2x)W_3$, `6 * seq_len * d_ff * (d_model + 1) = 1,007,262,105,600`, `32.6%`), which in total will be `3093372600320` and `148,481,884,815,360` for all layers. 
Besides transformer blocks, there are final norm (`4 * seq_len * d_model = 104,857,600`, `0.00007%`), LM head layer (`2 * vocab_size * d_model = 160,822,400`, `0.0001%`). 
In summary, `148,482,150,495,360` FLOPs are needed for the given sequence's forward pass, increased by **`313x`** from `460,463,924,736`. And the proportion taken by **Attention layer has grown dramatically**, while that of **previously dominant part FFN has shrunk**.
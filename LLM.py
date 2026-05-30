"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         LARGE LANGUAGE MODELS — COMPLETE ADVANCED GUIDE                        ║
║         From Design → Architecture → Training → Inference → Production         ║
║                                                                                ║
║  WHAT THIS FILE COVERS:                                                        ║
║  ─────────────────────────────────────────────────────────────                 ║
║  PART 0  : What LLMs are, their types, real-world uses                         ║
║  PART 1  : Tokenization (BPE from scratch)                                     ║
║  PART 2  : Token & Positional Embeddings (RoPE)                                ║
║  PART 3  : Scaled Dot-Product Attention                                        ║
║  PART 4  : Multi-Head Attention (MHA) + Causal Masking                        ║
║  PART 5  : Grouped Query Attention (GQA) — Llama 3 style                      ║
║  PART 6  : RMSNorm (Pre-LayerNorm)                                             ║
║  PART 7  : SwiGLU Feed-Forward Network                                         ║
║  PART 8  : Full Transformer Block (Pre-Norm + Residuals)                       ║
║  PART 9  : Complete GPT-style Decoder-Only LLM                                 ║
║  PART 10 : KV Cache for Inference                                              ║
║  PART 11 : Decoding Strategies (Greedy, Top-k, Top-p, Temperature)             ║
║  PART 12 : Training Loop (AdamW + Gradient Clipping + LR Scheduling)           ║
║  PART 13 : LoRA — Parameter-Efficient Fine-Tuning                              ║
║  PART 14 : RLHF Concepts (PPO reward structure)                                ║
║  PART 15 : Speculative Decoding                                                ║
║  PART 16 : Production — Quantization, Batching, Metrics                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import math
import copy
import random
import numpy as np
from collections import Counter, defaultdict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 0 ── WHAT IS AN LLM? TYPES & USES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
DEFINITION:
  A Large Language Model (LLM) is a neural network trained on massive text corpora
  to model the probability distribution P(token | context). During generation, it
  autoregressively samples one token at a time:

      P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)

  The "large" refers to both parameters (7B–1T+) and training data (trillions of tokens).

──────────────────────────────────────────────────────────────────────────
LLM TYPES (by architecture):
──────────────────────────────────────────────────────────────────────────

  1. DECODER-ONLY (autoregressive)  ← The dominant paradigm today
     - Examples: GPT-2/3/4, Llama 1/2/3, Mistral, Falcon, Gemma, Claude
     - Architecture: Causal transformer. Each token can only attend to past tokens.
     - Use cases: Text generation, chat, code, reasoning, few-shot learning
     - Training objective: Next-token prediction (CLM — Causal Language Modeling)

  2. ENCODER-ONLY (bidirectional)
     - Examples: BERT, RoBERTa, DeBERTa, ELECTRA
     - Architecture: Transformer encoder. Every token attends to ALL tokens.
     - Use cases: Classification, NER, semantic search, sentence similarity
     - Training objective: Masked Language Modeling (MLM) — predict masked tokens

  3. ENCODER-DECODER (seq2seq)
     - Examples: T5, BART, mT5, FLAN-T5, Pegasus
     - Architecture: Encoder processes input; Decoder generates output.
     - Use cases: Translation, summarization, question answering, data-to-text
     - Training objective: Reconstruction / conditional generation

──────────────────────────────────────────────────────────────────────────
LLM USES (by domain):
──────────────────────────────────────────────────────────────────────────
  Text generation   → Creative writing, blog posts, marketing copy
  Code generation   → GitHub Copilot, Cursor, Claude Code
  Reasoning         → Chain-of-thought, math problem solving (o1, DeepSeek-R1)
  RAG pipelines     → Document Q&A, knowledge bases, enterprise search
  Summarization     → Legal docs, medical notes, financial reports
  Translation       → Multilingual NLP, low-resource languages
  Classification    → Sentiment, intent detection, content moderation
  Embeddings        → Vector search, recommendations (text-embedding-3-large)
  Agents            → Tool use, web browsing, autonomous task execution

──────────────────────────────────────────────────────────────────────────
KEY DESIGN DECISIONS in modern LLMs:
──────────────────────────────────────────────────────────────────────────
  Tokenizer         → BPE (GPT), WordPiece (BERT), SentencePiece (T5, Llama)
  Norm placement    → Pre-LayerNorm (modern) vs Post-LayerNorm (original paper)
  Norm type         → RMSNorm (Llama) vs LayerNorm (GPT-2)
  Positional enc.   → RoPE (Llama, Mistral) vs ALiBi vs Learned (GPT-2)
  Activation        → SwiGLU (Llama) vs GELU (GPT-2) vs ReLU (original)
  Attention variant → GQA (Llama 3) vs MQA (Falcon) vs MHA (GPT-2)
  Context length    → 4k (Llama 1) → 8k → 128k (Llama 3) → 1M+ (Gemini 1.5)
"""

print("=" * 80)
print("  ADVANCED LLM FROM SCRATCH — NUMPY IMPLEMENTATION")
print("  Every component designed, explained, and runnable")
print("=" * 80)
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1 ── TOKENIZATION (BPE — Byte Pair Encoding)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
WHAT: Tokenization converts raw text → sequence of integer IDs.
      The model never sees characters or words — only integers.

WHY BPE:
  - Pure character tokenization → too many steps for long sequences.
  - Pure word tokenization → huge vocabulary, OOV (out-of-vocab) problem.
  - BPE finds a Goldilocks: common subwords are one token, rare words split.

HOW BPE WORKS:
  1. Start: vocabulary = all unique characters in training corpus.
  2. Count every pair of adjacent tokens in the corpus.
  3. Merge the most frequent pair into a new token. Repeat N times.
  4. "transformer" → ["t","r","a","n","s","f","o","r","m","e","r"]
     After learning: → ["trans", "##form", "##er"]  (3 tokens instead of 11)

  Real tokenizers (tiktoken for GPT, SentencePiece for Llama) run BPE on raw
  bytes — they never need a pre-tokenization step and handle ALL Unicode.

VOCABULARY SIZE in production:
  GPT-2: 50,257  |  Llama 3: 128,256  |  T5: 32,100
"""

class BPETokenizer:
    """
    Minimal BPE tokenizer built from scratch.
    Demonstrates the exact algorithm used by GPT-style models.
    """
    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.merges: dict[tuple, str] = {}   # merge rules: (a,b) → ab
        self.vocab: dict[str, int] = {}      # token string → integer ID
        self.id_to_token: dict[int, str] = {}

    def _get_pairs(self, word: list[str]) -> set[tuple]:
        """Return all adjacent pairs in a tokenized word."""
        return set(zip(word[:-1], word[1:]))

    def _merge_pair(self, pair: tuple, vocab_words: dict) -> dict:
        """Merge all occurrences of `pair` in every word."""
        a, b = pair
        new_vocab = {}
        for word, freq in vocab_words.items():
            tokens = list(word)
            i = 0
            merged = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            new_vocab[tuple(merged)] = freq
        return new_vocab

    def fit(self, texts: list[str]):
        """Learn BPE merges from a list of training strings."""
        # Step 1: Character-level vocabulary
        # Each word split into chars + </w> end-of-word marker
        word_freqs = Counter()
        for text in texts:
            for word in text.lower().split():
                word_freqs[word + "</w>"] += 1

        # Convert each word to tuple of chars
        vocab_words = {tuple(word): freq for word, freq in word_freqs.items()}

        # Step 2: Collect all unique characters as base vocabulary
        base_chars = set()
        for word in vocab_words:
            base_chars.update(word)

        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for ch in sorted(base_chars):
            self.vocab[ch] = len(self.vocab)

        # Step 3: BPE merge loop
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            # Count all adjacent pairs across all words
            pair_counts = Counter()
            for word, freq in vocab_words.items():
                pairs = self._get_pairs(list(word))
                for pair in pairs:
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Create new merged token
            new_token = "".join(best_pair)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = len(self.vocab)

            # Apply merge to all words
            vocab_words = self._merge_pair(best_pair, vocab_words)

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"[BPE] Trained. Vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> list[int]:
        """Apply learned BPE merges to encode text → IDs."""
        ids = [self.vocab.get("<BOS>", 2)]
        for word in text.lower().split():
            tokens = list(word + "</w>")
            # Apply all learned merges greedily
            changed = True
            while changed:
                changed = False
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1:
                        pair = (tokens[i], tokens[i+1])
                        if pair in self.merges:
                            new_tokens.append(self.merges[pair])
                            i += 2
                            changed = True
                            continue
                    new_tokens.append(tokens[i])
                    i += 1
                tokens = new_tokens
            for tok in tokens:
                ids.append(self.vocab.get(tok, self.vocab.get("<UNK>", 1)))
        ids.append(self.vocab.get("<EOS>", 3))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert IDs back to text."""
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids
                  if i not in (0, 2, 3)]  # skip PAD, BOS, EOS
        text = "".join(tokens).replace("</w>", " ").strip()
        return text


# Demo BPE
sample_corpus = [
    "the transformer learns context well",
    "transformers use attention mechanisms",
    "language models predict next tokens",
    "deep learning transforms natural language",
]
tokenizer = BPETokenizer(vocab_size=200)
tokenizer.fit(sample_corpus)

demo_text = "transformer learns language"
encoded = tokenizer.encode(demo_text)
decoded = tokenizer.decode(encoded)
print(f"\n[PART 1] Tokenization:")
print(f"  Input   : '{demo_text}'")
print(f"  Encoded : {encoded}")
print(f"  Decoded : '{decoded}'")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2 ── EMBEDDINGS + ROTARY POSITIONAL ENCODING (RoPE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
TOKEN EMBEDDING:
  A lookup table of shape [vocab_size, d_model].
  Each integer token ID maps to a dense vector of d_model floats.
  These vectors are LEARNED during training — they encode semantic meaning.

  Why d_model floats? Each dimension encodes some latent feature.
  Word2Vec showed similar words → similar vectors. LLMs learn richer versions.

POSITIONAL ENCODING — WHY IT'S NEEDED:
  Transformers process ALL tokens in parallel (unlike RNNs).
  Without position info, "dog bites man" == "man bites dog" to the model.

ROTARY POSITIONAL ENCODING (RoPE) — used in Llama, Mistral, GPT-NeoX:
  IDEA: Instead of adding a positional vector to the embedding, RoPE
        ROTATES the Q and K vectors in pairs of dimensions.

  For a pair of dimensions (q₂ᵢ, q₂ᵢ₊₁) at position m:
      q'₂ᵢ   = q₂ᵢ   · cos(m·θᵢ) − q₂ᵢ₊₁ · sin(m·θᵢ)
      q'₂ᵢ₊₁ = q₂ᵢ   · sin(m·θᵢ) + q₂ᵢ₊₁ · cos(m·θᵢ)
  where θᵢ = base^(−2i/d)  (base=10000 originally, 500000 in Llama 3)

  KEY PROPERTY: The dot product Q'·K' depends only on the RELATIVE
  distance (m−n), not absolute positions. This makes RoPE naturally
  generalize to longer sequences than seen during training (with YaRN/Scaling).

ALTERNATIVES:
  Learned absolute (GPT-2): just add a learned vector per position.
  ALiBi (BLOOM): subtract m·slope from attention score — no embedding change.
  Sinusoidal (original Transformer): fixed sin/cos, no learning.
"""

class TokenEmbedding:
    """Learned token embedding table."""
    def __init__(self, vocab_size: int, d_model: int):
        # Xavier uniform initialization: keeps variance stable across layers
        scale = math.sqrt(2.0 / (vocab_size + d_model))
        self.weight = np.random.uniform(-scale, scale, (vocab_size, d_model))
        print(f"[Embedding] Table: {vocab_size}×{d_model} = {vocab_size*d_model:,} params")

    def forward(self, ids: list[int]) -> np.ndarray:
        """Look up embedding vectors for token IDs. Returns [seq_len, d_model]."""
        return self.weight[ids]  # fancy indexing — O(seq_len)


def compute_rope_freqs(d_model: int, max_seq: int, base: float = 10000.0) -> np.ndarray:
    """
    Precompute RoPE rotation angles for all positions and dimension pairs.
    Returns [max_seq, d_model/2] matrix of angles.

    θᵢ = base^(−2i/d_model)  for i in [0, d_model/2)
    angle[m, i] = m · θᵢ
    """
    half_d = d_model // 2
    # Frequencies decrease exponentially — low-freq dims encode coarse position
    freqs = 1.0 / (base ** (np.arange(0, half_d) * 2 / d_model))
    positions = np.arange(max_seq)
    # Outer product: [max_seq, half_d]
    angles = np.outer(positions, freqs)
    return angles  # each entry = m · θᵢ


def apply_rope(x: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Apply RoPE rotation to Q or K tensor.
    x shape: [seq_len, d_model]
    angles shape: [seq_len, d_model/2]
    """
    seq_len, d_model = x.shape
    half_d = d_model // 2

    # Split into even/odd dimension pairs
    x1 = x[:, :half_d]   # even dims (x₂ᵢ)
    x2 = x[:, half_d:]   # odd dims  (x₂ᵢ₊₁)

    cos_a = np.cos(angles[:seq_len])
    sin_a = np.sin(angles[:seq_len])

    # Rotation formula applied element-wise
    rotated_x1 = x1 * cos_a - x2 * sin_a
    rotated_x2 = x1 * sin_a + x2 * cos_a

    return np.concatenate([rotated_x1, rotated_x2], axis=-1)


# Demo
d_model = 16
vocab_size = len(tokenizer.vocab)
emb = TokenEmbedding(vocab_size, d_model)
sample_ids = [2, 5, 8, 11, 3]  # BOS + 3 tokens + EOS
X = emb.forward(sample_ids)

rope_angles = compute_rope_freqs(d_model, max_seq=512)
X_rope = apply_rope(X, rope_angles)
print(f"\n[PART 2] Embedding + RoPE:")
print(f"  Token IDs      : {sample_ids}")
print(f"  After Embedding: shape {X.shape}")
print(f"  After RoPE     : shape {X_rope.shape}")
print(f"  Vector sample  : {X_rope[0, :4].round(4)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3 ── SCALED DOT-PRODUCT ATTENTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
THE CORE FORMULA:
    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

INTUITION step by step:
  1. Q (Query)   — "What am I looking for?" — shape [seq, d_k]
  2. K (Key)     — "What do I contain?"     — shape [seq, d_k]
  3. Q·Kᵀ        — Compatibility score between each query and every key
                   shape [seq_q, seq_k] — e.g. [6, 6]
  4. / √d_k      — Scale to prevent dot products from growing too large
                   (large values → extreme softmax → vanishing gradients)
                   WHY √d_k: random vectors of dim d_k have variance d_k,
                   so dividing by √d_k normalizes variance to 1.
  5. softmax()   — Convert scores to weights that sum to 1.
                   Token i's output = weighted sum of all Values.
  6. · V         — Attend to Value vectors proportionally to weights.
                   shape [seq_q, d_v]

CAUSAL MASK (for decoder-only LLMs):
  We add −∞ to the upper triangle of the score matrix BEFORE softmax.
  exp(−∞) = 0, so future tokens get zero weight.
  This ensures autoregressive property: position i only sees positions ≤ i.

FLASH ATTENTION (production optimization):
  Standard attention: materializes [seq, seq] matrix in HBM → O(n²) memory.
  FlashAttention: tiles Q,K,V into SRAM, fuses operations, never writes [n,n].
  Result: ~3x faster, O(n) memory, identical math.
"""

def scaled_dot_product_attention(
    Q: np.ndarray,          # [seq_q, d_k]
    K: np.ndarray,          # [seq_k, d_k]
    V: np.ndarray,          # [seq_k, d_v]
    mask: np.ndarray = None # [seq_q, seq_k] optional boolean mask
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full scaled dot-product attention.
    Returns (output [seq_q, d_v], attention_weights [seq_q, seq_k])
    """
    d_k = Q.shape[-1]

    # ── Step 1: Compatibility scores ─────────────────────────────────────
    # scores[i,j] = how much query i attends to key j
    scores = Q @ K.T / math.sqrt(d_k)    # [seq_q, seq_k]

    # ── Step 2: Apply causal mask ─────────────────────────────────────────
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # −1e9 ≈ −∞ before softmax

    # ── Step 3: Softmax (numerically stable) ─────────────────────────────
    # Subtract max per row before exp to prevent overflow (log-sum-exp trick)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)  # now sums to 1 per row

    # ── Step 4: Weighted sum of Values ────────────────────────────────────
    output = weights @ V                  # [seq_q, d_v]

    return output, weights


# Demo single-head attention
seq_len = 5
d_k = 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

# Causal mask: allow position i to see j ≤ i
causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

out, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"\n[PART 3] Scaled Dot-Product Attention:")
print(f"  Q,K,V shape : {Q.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Attn weights (row=query, col=key — upper triangle is 0):")
print(np.round(weights, 3))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 4 ── MULTI-HEAD ATTENTION (MHA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
WHY MULTIPLE HEADS?
  A single attention head looks at the sequence from ONE "perspective".
  Multiple heads run attention in PARALLEL with DIFFERENT weight matrices.

  Each head learns to specialize:
  - Head 1 might capture syntactic dependencies (subject-verb agreement)
  - Head 2 might track coreference ("it" → which entity?)
  - Head 3 might capture positional patterns (next word, nearby context)

HOW IT WORKS:
  1. Project input X into h sets of Q,K,V using separate weight matrices:
     Qᵢ = X·Wq_i   [seq, d_model] → [seq, d_k]   where d_k = d_model/h
     Kᵢ = X·Wk_i
     Vᵢ = X·Wv_i

  2. Run scaled dot-product attention for each head independently.
     headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ)    [seq, d_v]

  3. Concatenate all heads:
     MultiHead = Concat(head₁, ..., headₕ)   [seq, h·d_v] = [seq, d_model]

  4. Final linear projection:
     Output = MultiHead · Wo   [seq, d_model]

PARAMETER COUNT:
  Wq, Wk, Wv: each [d_model, d_model]  → 3 × d_model²
  Wo:         [d_model, d_model]        → d_model²
  Total:      4 × d_model²
  For d_model=4096: 67M params per attention layer!
"""

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads   # per-head key/query dimension
        self.d_v = d_model // num_heads   # per-head value dimension

        # Initialize weight matrices with Xavier uniform
        scale = math.sqrt(2.0 / (d_model + d_model))
        self.Wq = np.random.uniform(-scale, scale, (d_model, d_model))
        self.Wk = np.random.uniform(-scale, scale, (d_model, d_model))
        self.Wv = np.random.uniform(-scale, scale, (d_model, d_model))
        self.Wo = np.random.uniform(-scale, scale, (d_model, d_model))

        total_params = 4 * d_model ** 2
        print(f"[MHA] heads={num_heads}, d_k={self.d_k}, params={total_params:,}")

    def forward(
        self,
        x: np.ndarray,                  # [seq, d_model]
        rope_angles: np.ndarray = None, # [max_seq, d_k/2]
        mask: np.ndarray = None         # [seq, seq] causal mask
    ) -> np.ndarray:
        seq_len, _ = x.shape

        # ── Step 1: Linear projections ──────────────────────────────────
        Q = x @ self.Wq   # [seq, d_model]
        K = x @ self.Wk
        V = x @ self.Wv

        # ── Step 2: Apply RoPE to Q and K (not V) ───────────────────────
        # RoPE rotates each head's Q,K — since all heads share the same
        # positional signal but have different semantic projections
        if rope_angles is not None:
            # Reshape to [seq, h, d_k], apply RoPE per head, reshape back
            Q_heads = Q.reshape(seq_len, self.h, self.d_k)
            K_heads = K.reshape(seq_len, self.h, self.d_k)
            angles_head = compute_rope_freqs(self.d_k, max_seq=seq_len+64)
            for hi in range(self.h):
                Q_heads[:, hi, :] = apply_rope(Q_heads[:, hi, :], angles_head)
                K_heads[:, hi, :] = apply_rope(K_heads[:, hi, :], angles_head)
            Q = Q_heads.reshape(seq_len, self.d_model)
            K = K_heads.reshape(seq_len, self.d_model)

        # ── Step 3: Split into h heads ───────────────────────────────────
        # Reshape [seq, d_model] → [h, seq, d_k]
        Q = Q.reshape(seq_len, self.h, self.d_k).transpose(1, 0, 2)  # [h,seq,d_k]
        K = K.reshape(seq_len, self.h, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.h, self.d_v).transpose(1, 0, 2)

        # ── Step 4: Attend per head ──────────────────────────────────────
        head_outputs = []
        for hi in range(self.h):
            head_out, _ = scaled_dot_product_attention(Q[hi], K[hi], V[hi], mask)
            head_outputs.append(head_out)   # each [seq, d_v]

        # ── Step 5: Concatenate + output projection ──────────────────────
        concat = np.concatenate(head_outputs, axis=-1)  # [seq, d_model]
        output = concat @ self.Wo                        # [seq, d_model]
        return output


# Demo MHA
d_model = 32
num_heads = 4
seq_len = 6
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
x_demo = np.random.randn(seq_len, d_model)
causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
mha_out = mha.forward(x_demo, mask=causal_mask)
print(f"\n[PART 4] Multi-Head Attention:")
print(f"  Input  shape: {x_demo.shape}")
print(f"  Output shape: {mha_out.shape}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 5 ── GROUPED QUERY ATTENTION (GQA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
THE PROBLEM WITH MHA AT INFERENCE:
  During autoregressive generation, we must store K and V for every token
  in the sequence (the KV cache). For MHA with h heads:
    KV cache size = 2 × layers × h × seq_len × d_k × bytes
  For Llama 2 70B: 2 × 80 × 64 × 4096 × 8bytes = ~337GB per sequence!

SOLUTIONS:
  Multi-Query Attention (MQA): ALL query heads share ONE K and V head.
    - KV cache shrinks by factor of h
    - Quality degrades slightly (used in Falcon, early Starcoder)

  Grouped Query Attention (GQA): Query heads are split into G groups.
    Each group shares ONE K and V head.
    - g=1: same as MQA  |  g=h: same as MHA  |  g=2,4,8: sweet spot
    - Used in: Llama 2 (70B), Llama 3, Mistral, Gemma, Qwen

  Example: Llama 3 8B has 32 Q heads, 8 KV heads (g=4)
    → KV cache is 4x smaller vs MHA, almost no quality loss.

HOW GQA WORKS:
  - Project into n_q Q heads, n_kv K heads, n_kv V heads
  - For each KV head, its corresponding group of Q heads attend to it
  - Each Q head still sees full seq; only K,V are shared within groups
"""

class GroupedQueryAttention:
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        assert n_q_heads % n_kv_heads == 0, "Q heads must be divisible by KV heads"
        self.d_model = d_model
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.groups = n_q_heads // n_kv_heads  # Q heads per KV head
        self.d_k = d_model // n_q_heads

        scale = math.sqrt(1.0 / d_model)
        self.Wq = np.random.uniform(-scale, scale, (d_model, n_q_heads * self.d_k))
        self.Wk = np.random.uniform(-scale, scale, (d_model, n_kv_heads * self.d_k))
        self.Wv = np.random.uniform(-scale, scale, (d_model, n_kv_heads * self.d_k))
        self.Wo = np.random.uniform(-scale, scale, (n_q_heads * self.d_k, d_model))

        q_params = d_model * n_q_heads * self.d_k
        kv_params = 2 * d_model * n_kv_heads * self.d_k
        print(f"[GQA] Q-heads={n_q_heads}, KV-heads={n_kv_heads}, "
              f"groups={self.groups}, "
              f"KV-cache saving: {n_q_heads/n_kv_heads:.0f}x vs MHA")
        print(f"      Params: Q={q_params:,}, KV={kv_params:,}, "
              f"Total≈{q_params+kv_params+n_q_heads*self.d_k*d_model:,}")

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        seq_len, _ = x.shape

        # Project: more Q heads, fewer K/V heads
        Q = x @ self.Wq  # [seq, n_q * d_k]
        K = x @ self.Wk  # [seq, n_kv * d_k]
        V = x @ self.Wv  # [seq, n_kv * d_k]

        # Reshape to separate heads
        Q = Q.reshape(seq_len, self.n_q, self.d_k)   # [seq, n_q, d_k]
        K = K.reshape(seq_len, self.n_kv, self.d_k)  # [seq, n_kv, d_k]
        V = V.reshape(seq_len, self.n_kv, self.d_k)

        # For each KV head, attend with its group of Q heads
        head_outputs = []
        for kv_idx in range(self.n_kv):
            k_head = K[:, kv_idx, :]  # [seq, d_k]
            v_head = V[:, kv_idx, :]  # [seq, d_k]
            # Each Q head in this group attends to the SAME k,v
            for q_offset in range(self.groups):
                q_idx = kv_idx * self.groups + q_offset
                q_head = Q[:, q_idx, :]  # [seq, d_k]
                out, _ = scaled_dot_product_attention(q_head, k_head, v_head, mask)
                head_outputs.append(out)

        concat = np.concatenate(head_outputs, axis=-1)  # [seq, n_q * d_k]
        return concat @ self.Wo                          # [seq, d_model]


gqa = GroupedQueryAttention(d_model=32, n_q_heads=8, n_kv_heads=2)
gqa_out = gqa.forward(x_demo, mask=np.tril(np.ones((seq_len, seq_len), dtype=bool)))
print(f"\n[PART 5] GQA output shape: {gqa_out.shape}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 6 ── RMSNORM (Root Mean Square Normalization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
WHY NORMALIZATION?
  Without it, activations grow or shrink exponentially through N layers.
  Unstable training — gradients vanish (too small) or explode (too large).

LAYERNORM (original):
  For each token, normalize across d_model dimensions:
    μ = mean(x)    σ = std(x)
    LN(x) = γ · (x − μ) / (σ + ε) + β
  Learns γ (scale) and β (shift) per dimension.

RMSNORM (used in Llama, Gemma, Qwen):
  Drops the mean centering — only normalizes by RMS:
    RMS(x) = √( (1/d) · Σ xᵢ² )
    RMSNorm(x) = γ · x / RMS(x)
  WHY: 7-10% faster (no mean computation), similar quality.
  Observation: The mean centering in LayerNorm adds little benefit.

PRE-NORM vs POST-NORM:
  Original paper (Post-Norm): Norm applied AFTER attention/FFN + residual.
    → Less stable training for very deep models.
  Modern (Pre-Norm): Norm applied BEFORE attention/FFN.
    → More stable, allows training deeper models without warm-up issues.
    → Used in GPT-2 and all modern LLMs.
"""

class RMSNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)  # learned scale, initialized to 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [seq, d_model] → [seq, d_model]"""
        # RMS over last dimension (per token, across features)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / rms)


rms = RMSNorm(d_model=32)
x_unnorm = np.random.randn(4, 32) * 100   # large values, unstable
x_normed = rms.forward(x_unnorm)
print(f"\n[PART 6] RMSNorm:")
print(f"  Before RMS: mean={x_unnorm.mean():.1f}, std={x_unnorm.std():.1f}")
print(f"  After  RMS: mean={x_normed.mean():.4f}, std={x_normed.std():.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 7 ── SWIGLU FEED-FORWARD NETWORK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
WHY FFN AFTER ATTENTION?
  Attention is a routing mechanism — it aggregates information across positions.
  But it's linear (attention is a weighted average of Vs, then a linear projection).
  The FFN adds NON-LINEARITY and CAPACITY. Each token's representation is processed
  by a 2-layer MLP independently (same weights for all positions).

  The FFN stores FACTUAL KNOWLEDGE learned during training.
  (Research shows: remove FFN layers → model loses factual recall.)

CLASSIC FFN (GPT-2, original Transformer):
    FFN(x) = GELU( x·W₁ + b₁ ) · W₂ + b₂
    d_model → 4·d_model → d_model   (4x expansion ratio)

SwiGLU (Llama, PaLM, Gemma):
    FFN(x) = ( x·Wgate · Swish(x·Wup) ) · Wdown
    Three matrices instead of two.
    Swish(x) = x · sigmoid(x)
    WHY: GLU-style (Gated Linear Units) gating multiplies two paths together,
         allowing the network to suppress irrelevant features.
         Consistently outperforms GELU in perplexity benchmarks.

  d_model → (Wup: 4·d_model, Wgate: 4·d_model) → hadamard product → Wdown → d_model

EXPANSION RATIO:
  Typical: 4x (GPT-2), Llama uses ~2.7x (to compensate for the 3rd matrix).
  For d_model=4096: up-proj has 4096×11008 = 45M params!
  FFN params ≈ 2/3 of total model params in most architectures.
"""

def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation: x · σ(x). Smooth, non-monotonic."""
    return x / (1.0 + np.exp(-x))   # numerically stable sigmoid


class SwiGLUFFN:
    def __init__(self, d_model: int, expansion: float = 2.7):
        # Llama uses 2.7x expansion to keep param count similar to 4x classic FFN
        d_ff = int(d_model * expansion)
        # Round to multiple of 64 for hardware efficiency (CUDA tensor cores)
        d_ff = (d_ff + 63) // 64 * 64

        scale = math.sqrt(2.0 / (d_model + d_ff))
        self.W_up   = np.random.uniform(-scale, scale, (d_model, d_ff))  # up-projection
        self.W_gate = np.random.uniform(-scale, scale, (d_model, d_ff))  # gate
        self.W_down = np.random.uniform(-scale, scale, (d_ff, d_model))  # down-projection

        params = d_model * d_ff * 2 + d_ff * d_model
        print(f"[SwiGLU FFN] d_ff={d_ff}, params={params:,}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [seq, d_model] → [seq, d_model]"""
        # Up-project and gate in parallel
        up   = x @ self.W_up    # [seq, d_ff] — raw features
        gate = x @ self.W_gate  # [seq, d_ff] — gating signal

        # Hadamard product: gate controls which features pass through
        hidden = swish(gate) * up   # [seq, d_ff]

        # Down-project back to d_model
        return hidden @ self.W_down  # [seq, d_model]


ffn = SwiGLUFFN(d_model=32)
ffn_out = ffn.forward(x_demo)
print(f"\n[PART 7] SwiGLU FFN output shape: {ffn_out.shape}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 8 ── FULL TRANSFORMER BLOCK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
ONE TRANSFORMER BLOCK = Pre-Norm + Attention + Residual + Pre-Norm + FFN + Residual

RESIDUAL CONNECTIONS — critical insight:
  Output = LayerNorm(x) processed by attention/FFN, then ADDED back to x.
  x = x + Attention(RMSNorm(x))
  x = x + FFN(RMSNorm(x))

  WHY: If attention/FFN output is all zeros, the block is identity function.
       The network learns CORRECTIONS to the residual stream, not the full transformation.
       This makes training much more stable — gradient highway through residuals.

THE RESIDUAL STREAM VIEW (modern interpretation):
  Think of x as a "residual stream" flowing through all layers.
  Each block reads from the stream, computes a delta, writes it back.
  Attention and FFN are parallel writers that modify the stream.
  This explains why middle layers' outputs can be deleted with minimal impact —
  most blocks write small corrections; only a few write large updates.

FULL DATA FLOW:
  input: x [seq, d_model]
    │
    ├── RMSNorm → MHA → + x  (attention sublayer with residual)
    │
    └── RMSNorm → FFN → + x  (FFN sublayer with residual)
    │
  output: x [seq, d_model]  (same shape — blocks are residual)
"""

class TransformerBlock:
    def __init__(self, d_model: int, num_heads: int, n_kv_heads: int = None):
        n_kv = n_kv_heads or num_heads  # default to full MHA
        self.norm1 = RMSNorm(d_model)
        self.attn  = GroupedQueryAttention(d_model, num_heads, n_kv)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # ── Attention sublayer (Pre-Norm + Residual) ─────────────────────
        x_norm = self.norm1.forward(x)
        attn_out = self.attn.forward(x_norm, mask=mask)
        x = x + attn_out              # Residual connection #1

        # ── FFN sublayer (Pre-Norm + Residual) ───────────────────────────
        x_norm = self.norm2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        x = x + ffn_out               # Residual connection #2

        return x


block = TransformerBlock(d_model=32, num_heads=8, n_kv_heads=2)
block_out = block.forward(x_demo, mask=np.tril(np.ones((seq_len, seq_len), dtype=bool)))
print(f"\n[PART 8] Transformer Block:")
print(f"  Input  shape: {x_demo.shape}")
print(f"  Output shape: {block_out.shape}")
print(f"  Input == Output shape: {x_demo.shape == block_out.shape} (residual property)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 9 ── COMPLETE GPT-STYLE DECODER-ONLY LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
FULL ARCHITECTURE:
  Embedding(input_ids)             [batch, seq] → [seq, d_model]
  + RoPE positional info
  → N × TransformerBlock           [seq, d_model] → [seq, d_model]
  → Final RMSNorm                  [seq, d_model]
  → LM Head (linear, no bias)     [seq, d_model] → [seq, vocab_size]
  → Logits → Softmax (at inference)

LM HEAD (unembedding):
  The LM head projects the final hidden state to vocabulary logits.
  In most models it SHARES WEIGHTS with the token embedding (weight tying):
    LM_head.weight = Embedding.weight.T
  This saves vocab_size × d_model parameters and often improves quality
  because the model learns consistent representations for input and output.

PARAMETER SCALING (rough formulas):
  Attention params per layer: 4 × d_model²
  FFN params per layer:       3 × d_model × d_ff  (SwiGLU)
  Total params ≈ N × (4 × d_model² + 3 × d_model × d_ff)
               + vocab_size × d_model  (embedding/LM head)

  Llama 3 8B:  d=4096, N=32, h=32, kv_h=8  → ~8B params
  Llama 3 70B: d=8192, N=80, h=64, kv_h=8  → ~70B params
  GPT-4:       ~1.8T params (mixture of experts, 8×220B experts)

TRAINING OBJECTIVE — Next Token Prediction (Causal LM):
  For each position i, predict token i+1 given tokens 0..i.
  Loss = cross-entropy averaged over all positions:
    L = −(1/T) Σᵢ log P(xᵢ₊₁ | x₀..xᵢ)
  This is also called "teacher forcing" — during training, ground-truth
  tokens are always fed as input (even if the model's prediction was wrong).
"""

class DecoderOnlyLLM:
    """
    Complete GPT/Llama-style decoder-only transformer.
    Implements: embedding → N transformer blocks → final norm → LM head.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 512,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        print(f"\n[LLM] Building model:")
        print(f"  vocab={vocab_size}, d_model={d_model}, layers={num_layers}")
        print(f"  heads={num_heads}, kv_heads={n_kv_heads}, max_seq={max_seq_len}")

        # Token embedding
        self.embedding = TokenEmbedding(vocab_size, d_model)

        # Precompute RoPE angles for all positions
        self.rope_angles = compute_rope_freqs(d_model // num_heads, max_seq_len)

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, n_kv_heads)
            for _ in range(num_layers)
        ]

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # LM head — weight-tied with embedding
        # In practice: lm_head.weight = embedding.weight (same array)
        # Here we keep separate for clarity but note the tie
        self.lm_head = self.embedding.weight  # [vocab_size, d_model]

        # Estimate parameter count
        attn_per_layer = 4 * d_model * d_model  # rough (ignores GQA reduction)
        ffn_per_layer  = 3 * d_model * int(d_model * 2.7)
        total = (attn_per_layer + ffn_per_layer) * num_layers + vocab_size * d_model
        print(f"  ~{total/1e6:.1f}M parameters")

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """
        Forward pass: token IDs → logits over vocabulary.
        Returns [seq_len, vocab_size] logits.
        """
        seq_len = len(token_ids)

        # ── Step 1: Token Embedding ────────────────────────────────────────
        x = self.embedding.forward(token_ids)   # [seq, d_model]

        # ── Step 2: Build causal mask ─────────────────────────────────────
        # Lower triangular: position i can only attend to positions ≤ i
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

        # ── Step 3: Run through all transformer blocks ────────────────────
        for block in self.blocks:
            x = block.forward(x, mask=causal_mask)

        # ── Step 4: Final normalization ───────────────────────────────────
        x = self.final_norm.forward(x)           # [seq, d_model]

        # ── Step 5: Project to vocabulary logits ──────────────────────────
        # lm_head is [vocab_size, d_model], so: x @ lm_head.T
        logits = x @ self.lm_head.T              # [seq, vocab_size]

        return logits

    def get_next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        """Return only the logits for the LAST position (next-token prediction)."""
        logits = self.forward(token_ids)
        return logits[-1]   # [vocab_size]


# Build tiny LLM (realistic architecture, small dims for demo)
llm = DecoderOnlyLLM(
    vocab_size=vocab_size,
    d_model=32,
    num_layers=2,
    num_heads=8,
    n_kv_heads=2,
    max_seq_len=128,
)

demo_ids = tokenizer.encode("transformer learns")
logits = llm.get_next_token_logits(demo_ids)
print(f"\n[PART 9] Full LLM forward pass:")
print(f"  Input IDs  : {demo_ids}")
print(f"  Logit shape: {logits.shape}  (one score per vocab token)")
print(f"  Top-5 token IDs: {np.argsort(logits)[-5:][::-1]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 10 ── KV CACHE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
THE PROBLEM WITHOUT KV CACHE:
  Autoregressive generation: to produce token T+1, we feed tokens [0..T].
  For token T+2: we feed [0..T+1] — computing K,V for tokens 0..T AGAIN.
  → Quadratic cost O(T²) for generating T tokens. Unacceptable for long seqs.

KV CACHE SOLUTION:
  After computing K and V for token i, SAVE them.
  When generating token T+1, only compute Q for the NEW token (position T).
  Attend using: saved K[0..T], saved V[0..T], new Q[T].
  → Linear cost O(T) per new token. Constant amortized inference.

MEMORY COST:
  KV cache size = 2 × layers × n_kv_heads × seq_len × d_k × dtype_bytes
  For Llama 3 8B (BF16): 2 × 32 × 8 × 4096 × 128 × 2 bytes = 268MB per sequence
  For batch_size=32: 8.6GB just for KV cache!

OPTIMIZATIONS:
  - Paged Attention (vLLM): KV cache is stored in non-contiguous pages,
    no pre-allocation needed — enables dynamic batch sizes.
  - Multi-query + GQA: fewer KV heads → proportionally less cache.
  - Quantized KV cache: store K,V in INT8 or FP8 → 2-4x less memory.

PREFILL vs DECODE phase:
  Prefill: process full prompt in one parallel forward pass. O(prompt² / 2).
  Decode: one token at a time, using KV cache. O(n) per step.
  This is why long prompts are processed faster than long generations.
"""

class KVCache:
    """
    Simple KV cache for a single attention layer.
    In production: stored on GPU HBM, pre-allocated per max_seq_len.
    """
    def __init__(self):
        self.k_cache: list[np.ndarray] = []   # list of [1, d_k] per step
        self.v_cache: list[np.ndarray] = []

    def update(self, k_new: np.ndarray, v_new: np.ndarray):
        """Append new K,V for the latest token."""
        self.k_cache.append(k_new)
        self.v_cache.append(v_new)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """Return full cached K and V."""
        return (
            np.concatenate(self.k_cache, axis=0),   # [cached_len, d_k]
            np.concatenate(self.v_cache, axis=0),
        )

    @property
    def length(self) -> int:
        return len(self.k_cache)


# Demonstrate KV cache concept
cache = KVCache()
d_k = 8
print(f"\n[PART 10] KV Cache demonstration:")
for step in range(4):
    # New token produces one K,V vector
    k_new = np.random.randn(1, d_k)
    v_new = np.random.randn(1, d_k)
    cache.update(k_new, v_new)
    K_full, V_full = cache.get()
    print(f"  Step {step+1}: new Q attends to {cache.length} cached K/V vectors "
          f"(K shape: {K_full.shape})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 11 ── DECODING STRATEGIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
WHAT IS DECODING?
  The model outputs a distribution over vocabulary at each step.
  Decoding decides WHICH token to pick from that distribution.
  Different strategies trade off quality, diversity, and determinism.

1. GREEDY DECODING
   Always pick the highest-probability token.
   → Deterministic, fast, but repetitive ("I think I think I think...")
   → No randomness → model gets stuck in loops.

2. TEMPERATURE SCALING
   Divide logits by T before softmax:  softmax(logits / T)
   T < 1.0: Distribution sharpens → more confident, less creative
   T = 1.0: Original distribution (no change)
   T > 1.0: Distribution flattens → more random, more diverse
   T → 0:   Equivalent to greedy  |  T → ∞: Uniform random
   Used for: T=0.7 for factual, T=1.0 for creative writing

3. TOP-K SAMPLING
   Keep only the K highest probability tokens, re-normalize, sample.
   Common values: k=50 (GPT-2 default), k=40, k=100
   Problem: K is fixed regardless of distribution shape.
     If distribution is very peaked (1 obvious token), K=50 still samples junk.

4. TOP-P (NUCLEUS) SAMPLING  ← most commonly used in production
   Keep smallest set of tokens whose cumulative probability ≥ p.
   Dynamic K: flat distribution → include many tokens; peaked → include few.
   Common values: p=0.9, p=0.95
   Proposed by Holtzman et al. 2020 "The Curious Case of Neural Text Degeneration"

5. REPETITION PENALTY
   For tokens already in the context, divide their logits by penalty factor.
   penalty=1.0: no effect  |  penalty=1.3: moderately penalizes repeats
   Prevents: "I love cats. I love cats. I love cats."

6. BEAM SEARCH
   Maintain B candidate sequences ("beams") at each step.
   Expand each beam by all vocab tokens, keep top-B sequences by score.
   → Better quality for translation/summarization, poor for open-ended chat
   → Quadratic memory and compute with beam width B.
"""

class DecodingStrategies:

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()

    @staticmethod
    def greedy(logits: np.ndarray) -> int:
        """Pick argmax — deterministic, fastest."""
        return int(np.argmax(logits))

    @staticmethod
    def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
        """Scale logits by temperature then sample."""
        if temperature <= 0:
            return DecodingStrategies.greedy(logits)
        scaled = logits / temperature
        probs = DecodingStrategies.softmax(scaled)
        return int(np.random.choice(len(probs), p=probs))

    @staticmethod
    def top_k_sample(logits: np.ndarray, k: int = 50, temperature: float = 1.0) -> int:
        """Keep top-k tokens, re-normalize, sample."""
        # Get k largest indices
        top_k_idx = np.argsort(logits)[-k:]
        top_k_logits = logits[top_k_idx]

        # Apply temperature
        top_k_logits = top_k_logits / temperature
        probs = DecodingStrategies.softmax(top_k_logits)

        # Sample from the top-k
        chosen_local = int(np.random.choice(k, p=probs))
        return int(top_k_idx[chosen_local])

    @staticmethod
    def top_p_sample(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
        """Nucleus sampling: smallest set of tokens with cumulative prob ≥ p."""
        # Sort descending
        sorted_idx = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_idx]

        # Apply temperature
        scaled = sorted_logits / temperature
        probs = DecodingStrategies.softmax(scaled)

        # Compute cumulative probs
        cumprobs = np.cumsum(probs)

        # Find nucleus: tokens needed to reach probability mass p
        # +1 to always include at least 1 token
        nucleus_size = int(np.sum(cumprobs <= p)) + 1
        nucleus_size = min(nucleus_size, len(probs))

        nucleus_idx = sorted_idx[:nucleus_size]
        nucleus_probs = probs[:nucleus_size]
        nucleus_probs /= nucleus_probs.sum()   # re-normalize to sum to 1

        chosen_local = int(np.random.choice(nucleus_size, p=nucleus_probs))
        return int(nucleus_idx[chosen_local])

    @staticmethod
    def apply_repetition_penalty(
        logits: np.ndarray,
        input_ids: list[int],
        penalty: float = 1.3
    ) -> np.ndarray:
        """Divide logits of already-seen tokens by penalty factor."""
        logits = logits.copy()
        for token_id in set(input_ids):
            if 0 <= token_id < len(logits):
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty
        return logits


# Demonstrate decoding
np.random.seed(42)
demo_logits = np.random.randn(vocab_size)

greedy_tok = DecodingStrategies.greedy(demo_logits)
topk_tok   = DecodingStrategies.top_k_sample(demo_logits, k=10, temperature=0.8)
topp_tok   = DecodingStrategies.top_p_sample(demo_logits, p=0.9, temperature=0.9)

print(f"\n[PART 11] Decoding Strategies (from logits over {vocab_size} tokens):")
print(f"  Greedy pick   : token {greedy_tok} = '{tokenizer.id_to_token.get(greedy_tok, '?')}'")
print(f"  Top-K (k=10)  : token {topk_tok}  = '{tokenizer.id_to_token.get(topk_tok, '?')}'")
print(f"  Top-P (p=0.9) : token {topp_tok}  = '{tokenizer.id_to_token.get(topp_tok, '?')}'")

# Full generation loop
def generate(
    model: DecoderOnlyLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 0.9,
    top_p: float = 0.9,
) -> str:
    """Autoregressive token generation loop."""
    ids = tokenizer.encode(prompt)
    eos_id = tokenizer.vocab.get("<EOS>", 3)

    for _ in range(max_new_tokens):
        # Forward pass: get logits for next token
        next_logits = model.get_next_token_logits(ids)

        # Apply repetition penalty
        next_logits = DecodingStrategies.apply_repetition_penalty(
            next_logits, ids, penalty=1.2
        )

        # Sample next token using top-p
        next_id = DecodingStrategies.top_p_sample(
            next_logits, p=top_p, temperature=temperature
        )

        # Stop if EOS
        if next_id == eos_id:
            break

        ids.append(next_id)

    return tokenizer.decode(ids)


generated = generate(llm, tokenizer, "transformer learns", max_new_tokens=6)
print(f"\n  Generated text: '{generated}' (random weights — for structure demo)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 12 ── TRAINING LOOP (AdamW + Gradient Clipping + LR Scheduling)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
TRAINING OBJECTIVE: Next Token Prediction (Causal Language Modeling)
  For sequence [x₀, x₁, ..., xₜ]:
    loss = −(1/T) Σᵢ log P(xᵢ₊₁ | x₀..xᵢ)
  = average cross-entropy over all next-token predictions.

  PERPLEXITY = exp(loss) — interpretable metric.
    PPL=10 means model is as uncertain as picking uniformly from 10 tokens.
    GPT-2 small: ~29.4 PPL on WikiText-103. Llama 3 8B: ~6.1 PPL.

ADAMW OPTIMIZER:
  The standard choice for LLM training (used by GPT, Llama, everything).

  Adam adds momentum (first moment m) and adaptive learning rates (second moment v):
    m = β₁·m + (1−β₁)·g      (exponential moving average of gradients)
    v = β₂·v + (1−β₂)·g²     (EMA of squared gradients)
    m̂ = m / (1−β₁ᵗ)          (bias correction for first steps)
    v̂ = v / (1−β₂ᵗ)
    θ = θ − lr · m̂ / (√v̂ + ε)

  "W" in AdamW: Weight Decay is applied DIRECTLY to weights, NOT to gradients.
  (Regular Adam's L2 regularization interacts badly with adaptive LR.)
    θ = θ − lr · (m̂/(√v̂+ε) + λ·θ)
  Typical: β₁=0.9, β₂=0.95 (not 0.999), ε=1e-8, λ=0.1

GRADIENT CLIPPING:
  LLM training can have sudden gradient spikes ("loss spikes").
  Clip the global gradient norm to max_norm (typically 1.0):
    if ||g|| > max_norm: g = g × (max_norm / ||g||)
  Prevents a single bad batch from destroying trained weights.

LEARNING RATE SCHEDULING:
  1. Warmup: LR linearly increases from 0 to peak over first N steps.
     WHY: At init, parameters are random — large LR causes chaotic updates.
     Typical: warmup_steps = 2000
  2. Cosine decay: after warmup, LR follows cosine curve to lr_min.
     lr(t) = lr_min + 0.5·(lr_max−lr_min)·(1 + cos(π·t/T_max))
  3. Some models add a constant "hold" phase or use linear decay.

TYPICAL HYPERPARAMS (Llama 3 8B training):
  lr = 3e-4, warmup_steps=2000, β₁=0.9, β₂=0.95, ε=1e-5
  weight_decay=0.1, grad_clip=1.0, batch=4M tokens per step
  Total: 15T tokens of training data!
"""

class AdamW:
    """AdamW optimizer — the standard for LLM training."""
    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.95,    # 0.95 for LLMs, not 0.999
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.t = 0

        # Moment buffers — same shape as each parameter
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray], lr_override: float = None):
        """One AdamW update step given gradients."""
        self.t += 1
        lr = lr_override or self.lr

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update biased first/second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # Bias-corrected moments (critical for early steps)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # AdamW update: weight decay applied to param, not gradient
            param -= lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * param)


def cosine_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5,
) -> float:
    """Warmup + cosine annealing schedule."""
    if step < warmup_steps:
        # Linear warmup from 0 to lr_max
        return lr_max * step / warmup_steps
    else:
        # Cosine decay from lr_max to lr_min
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def clip_grad_norm(grads: list[np.ndarray], max_norm: float = 1.0) -> float:
    """Global gradient norm clipping. Returns the pre-clip norm."""
    global_norm = math.sqrt(sum(np.sum(g**2) for g in grads))
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-6)
        for g in grads:
            g *= scale
    return global_norm


def cross_entropy_loss(logits: np.ndarray, targets: list[int]) -> float:
    """
    Cross-entropy loss for next-token prediction.
    logits: [seq, vocab] — model predictions
    targets: [seq] — ground truth next tokens
    Loss is averaged over all positions.
    """
    total_loss = 0.0
    seq_len = len(targets)
    for i, target_id in enumerate(targets):
        # Softmax of logits at position i
        log_probs = logits[i] - np.log(np.sum(np.exp(logits[i] - logits[i].max())) + 1e-9)
        log_probs -= logits[i].max()  # numerical stability
        probs = np.exp(logits[i] - logits[i].max())
        probs /= probs.sum()
        # Negative log probability of correct token
        total_loss -= math.log(probs[target_id] + 1e-9)
    return total_loss / seq_len


# Demo training schedule
print(f"\n[PART 12] Training Components:")
steps = [0, 100, 500, 1000, 2000, 5000, 10000]
warmup, total = 2000, 10000
for s in steps:
    lr = cosine_lr_schedule(s, warmup, total)
    print(f"  step {s:5d}: lr = {lr:.2e}")

# Mini training step simulation
train_text = "transformer learns language models predict tokens"
ids = tokenizer.encode(train_text)
if len(ids) > 1:
    input_ids = ids[:-1]    # feed all but last
    target_ids = ids[1:]    # predict all but first (shifted by 1)
    fwd_logits = llm.forward(input_ids)
    loss = cross_entropy_loss(fwd_logits, target_ids)
    print(f"\n  Train loss example: {loss:.4f}")
    print(f"  Perplexity: {math.exp(loss):.2f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 13 ── LoRA: LOW-RANK ADAPTATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
THE FINE-TUNING PROBLEM:
  Full fine-tuning a 70B model requires:
  - 140GB+ of GPU memory just for weights (FP16)
  - Same amount again for optimizer states (AdamW uses 3x model size)
  - Large gradient tensors
  Total: ~560GB → requires 8× A100 80GB just for fine-tuning.

LoRA — Low-Rank Adaptation (Hu et al., 2021):
  KEY INSIGHT: LLM weight matrices W have low intrinsic rank for most tasks.
  When fine-tuning, the weight UPDATE ΔW also has low rank.

  Instead of learning full ΔW ∈ ℝ^(d×k), learn:
    ΔW = A × B    where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k),   r << min(d,k)

  Forward pass with LoRA:
    y = x·(W + ΔW) = x·W + x·A·B = original_output + lora_output

  PARAMETERS:
    Original W: d×k parameters (frozen, not updated)
    LoRA A,B:   d×r + r×k = r(d+k) parameters (trainable)
    Savings ratio: r(d+k) / (d×k) = r/d + r/k ≈ 2r/d for square matrices

  Example: d=k=4096, r=16
    Original: 16.7M params  |  LoRA: 16 × 8192 = 131K params  |  0.78% of original!

HYPERPARAMETERS:
  r (rank): higher rank = more capacity. Typical: 8, 16, 64
  alpha: scaling factor for ΔW. Effective scale = alpha/r.
         Often set alpha=2r so scale=2 regardless of r.
  target_modules: which weight matrices to apply LoRA to.
    Typically: Wq, Wk, Wv, Wo (attention projections)
    Sometimes also: up-proj, down-proj in FFN

QLoRA (Dettmers et al., 2023):
  = LoRA + 4-bit NormalFloat quantization of base weights
  → Fine-tune a 65B model on a SINGLE 48GB GPU!
  Base weights stored in NF4, dequantized to BF16 for forward pass.

SCALING:
  At merge time: W_merged = W + (alpha/r) × A × B
  Then discard A and B — inference is identical to original model speed.
"""

class LoRALayer:
    """
    LoRA adapter for a single linear layer.
    Freezes W, adds trainable low-rank A,B matrices.
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 8, alpha: float = 16.0):
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank   # effective learning rate scaling

        # Base weight — FROZEN during LoRA fine-tuning
        scale_w = math.sqrt(2.0 / (d_in + d_out))
        self.W = np.random.uniform(-scale_w, scale_w, (d_in, d_out))

        # LoRA matrices — TRAINABLE
        # A initialized with Gaussian, B initialized with zeros
        # (So ΔW = A×B = 0 at start → model starts = original pretrained model)
        self.A = np.random.randn(d_in, rank) * 0.02   # small Gaussian init
        self.B = np.zeros((rank, d_out))               # zero init → ΔW=0 initially

        trainable = d_in * rank + rank * d_out
        total = d_in * d_out
        print(f"[LoRA] Layer ({d_in}×{d_out}): rank={rank}, "
              f"trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward with LoRA:
          y = x·W  +  (alpha/r) · x·A·B
          ───base───   ──────lora delta────
        """
        base_out = x @ self.W                        # original computation
        lora_out = (x @ self.A) @ self.B * self.scale  # low-rank delta

        return base_out + lora_out

    def merge_weights(self) -> np.ndarray:
        """
        Merge LoRA weights into W (for efficient inference).
        After merge: W' = W + (alpha/r) · A·B
        LoRA A,B can be discarded — inference is now identical to baseline.
        """
        merged = self.W + self.scale * (self.A @ self.B)
        return merged


# Demonstrate LoRA
d_in, d_out = 32, 32
lora = LoRALayer(d_in, d_out, rank=4, alpha=8.0)

test_input = np.random.randn(5, d_in)
lora_out = lora.forward(test_input)
W_merged = lora.merge_weights()

# Verify: merged forward = original + LoRA delta
merged_out = test_input @ W_merged
print(f"\n[PART 13] LoRA:")
print(f"  LoRA output shape: {lora_out.shape}")
print(f"  Max diff (lora.forward vs merged): "
      f"{np.abs(lora_out - merged_out).max():.2e}  (should be ~0)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 14 ── RLHF: REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
WHY RLHF?
  Pre-training objective (next-token prediction) optimizes for fitting the
  data distribution. The model learns to complete text like the internet.
  But we want a model that is: Helpful, Harmless, Honest (Anthropic's 3H).

  The internet contains: conspiracy theories, toxic content, incorrect facts,
  repetitive boilerplate. A purely pre-trained model inherits all of this.

RLHF PIPELINE (InstructGPT, ChatGPT, Claude):

  STEP 1 — SUPERVISED FINE-TUNING (SFT):
    Human annotators write (prompt, ideal_response) pairs.
    Fine-tune the base model on these with standard cross-entropy.
    Result: Model that follows instructions, but not yet well-calibrated.

  STEP 2 — REWARD MODEL (RM) TRAINING:
    For each prompt, collect two model responses A and B.
    Human annotator ranks: "A is better because X".
    Train a reward model: given (prompt, response) → scalar reward score.
    RM uses same transformer architecture, adds a scalar head on top.
    Trained with Bradley-Terry loss:
      loss = −log σ(reward_winner − reward_loser)

  STEP 3 — PPO (Proximal Policy Optimization):
    Use RM as reward signal to fine-tune the SFT model with RL.
    The LLM is the POLICY: given prompt → generate response.
    Reward = RM(response) − β·KL(policy || reference_SFT)
    KL penalty prevents model from drifting too far from SFT (reward hacking).

    PPO objective (clipped):
      L = E[min(r(θ)·A, clip(r(θ), 1−ε, 1+ε)·A)]
    where r(θ) = π_θ(a|s) / π_ref(a|s) is the probability ratio.

  ALTERNATIVES TO PPO:
    DPO (Direct Preference Optimization): Skips RM, trains directly on
        preference pairs. Simpler, no RL instability. Used in Llama 2/3.
        loss = −log σ(β · log(π_θ(yw)/π_ref(yw)) − β · log(π_θ(yl)/π_ref(yl)))

    RLAIF (Constitutional AI, Anthropic): Use another LLM as the "human" rater.
        Much cheaper to scale. Used in Claude's training.

    GRPO (Group Relative Policy Optimization): Used in DeepSeek-R1.
        No reference model needed, estimates baseline from group of rollouts.
"""

class RewardModel:
    """
    Conceptual reward model for RLHF.
    In practice: same architecture as LLM, adds linear head on [CLS] token.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        # Single linear head: d_model → 1 scalar reward
        self.reward_head = np.random.randn(d_model) * 0.01

    def score(self, hidden_state: np.ndarray) -> float:
        """
        hidden_state: [d_model] — last token's hidden state
        Returns: scalar reward score
        """
        return float(self.hidden_state_to_scalar(hidden_state))

    def hidden_state_to_scalar(self, h: np.ndarray) -> float:
        return float(h @ self.reward_head)

    def bradley_terry_loss(
        self,
        h_chosen: np.ndarray,    # hidden state for chosen (better) response
        h_rejected: np.ndarray,  # hidden state for rejected response
    ) -> float:
        """
        Bradley-Terry preference loss.
        Maximizes probability that chosen response scores higher.
        loss = −log σ(r_chosen − r_rejected)
        """
        r_chosen   = self.hidden_state_to_scalar(h_chosen)
        r_rejected = self.hidden_state_to_scalar(h_rejected)
        margin = r_chosen - r_rejected
        # Sigmoid of margin = P(chosen > rejected) under Bradley-Terry model
        loss = -math.log(1.0 / (1.0 + math.exp(-margin)) + 1e-8)
        return loss


def dpo_loss(
    logp_chosen_policy: float,    # log P_θ(chosen | prompt)
    logp_rejected_policy: float,  # log P_θ(rejected | prompt)
    logp_chosen_ref: float,       # log P_ref(chosen | prompt)
    logp_rejected_ref: float,     # log P_ref(rejected | prompt)
    beta: float = 0.1,
) -> float:
    """
    Direct Preference Optimization loss.
    No reward model needed! Directly optimizes for preference.
    β controls how much the policy can deviate from reference.
    """
    # Implicit reward = β · log(π_θ / π_ref)
    chosen_reward   = beta * (logp_chosen_policy - logp_chosen_ref)
    rejected_reward = beta * (logp_rejected_policy - logp_rejected_ref)

    # Loss: −log σ(implicit_reward_margin)
    margin = chosen_reward - rejected_reward
    loss = -math.log(1.0 / (1.0 + math.exp(-margin)) + 1e-8)
    return loss


# Demo RLHF concepts
d_model = 32
rm = RewardModel(d_model)

h_chosen   = np.random.randn(d_model)
h_rejected = np.random.randn(d_model)

bt_loss = rm.bradley_terry_loss(h_chosen, h_rejected)

# DPO demo
dpo = dpo_loss(
    logp_chosen_policy=-1.2,
    logp_rejected_policy=-2.5,
    logp_chosen_ref=-1.5,
    logp_rejected_ref=-2.3,
    beta=0.1,
)

print(f"\n[PART 14] RLHF Concepts:")
print(f"  Reward (chosen)  : {rm.hidden_state_to_scalar(h_chosen):.4f}")
print(f"  Reward (rejected): {rm.hidden_state_to_scalar(h_rejected):.4f}")
print(f"  Bradley-Terry loss: {bt_loss:.4f}")
print(f"  DPO loss          : {dpo:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 15 ── SPECULATIVE DECODING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
THE BOTTLENECK: LLM inference is memory-bandwidth bound, not compute bound.
  Loading 70B weight parameters from HBM for each token takes ~milliseconds.
  The GPU is mostly WAITING for weights to arrive, not doing math.

SPECULATIVE DECODING (Leviathan et al., 2022):
  Use a SMALL fast "draft" model to generate K candidate tokens quickly.
  Then use the LARGE target model to VERIFY all K tokens in one parallel pass.
  Tokens that match the target distribution are ACCEPTED; first mismatch → reject.

  KEY PROPERTY: Speculative decoding is MATHEMATICALLY EQUIVALENT to sampling
  from the target model distribution — no quality loss, only speed gain!

ACCEPTANCE CRITERION (rejection sampling):
  Draft proposes token x with probability q(x).
  Target model would choose x with probability p(x).
  Accept x if q(x) ≤ p(x), i.e., the draft was "too confident" about x.
  Otherwise accept with probability 1 − p(x)/q(x) and resample from target.

WHY IT'S FASTER:
  Target model's GPU pass is the same cost whether processing 1 or K tokens.
  (Memory-bandwidth bottleneck = same weight-loading cost for batch size 1 vs K)
  If draft model accepts rate is α (typically 70-90%), we get ~K·α new tokens
  per target-model evaluation instead of 1.
  Typical speedup: 2-3x for K=4-8 draft tokens.

DRAFT MODEL CHOICES:
  - Smaller version of same family (Llama 3 8B drafts for Llama 3 70B)
  - Tiny dedicated draft model (Medusa heads — multiple heads on target itself)
  - Self-speculation (prompt lookup decoding — repeat common ngrams)
"""

def speculative_decode_step(
    draft_probs: np.ndarray,    # [K, vocab] — draft token probabilities
    target_probs: np.ndarray,   # [K+1, vocab] — target model probabilities
    draft_tokens: list[int],    # K draft token IDs
    temperature: float = 1.0,
) -> tuple[list[int], int]:
    """
    One round of speculative decoding.
    Returns (accepted_tokens, bonus_token_from_target).
    Guaranteed to be equivalent to sampling from target distribution.
    """
    accepted = []

    for k, token_id in enumerate(draft_tokens):
        q = draft_probs[k, token_id]   # draft prob for this token
        p = target_probs[k, token_id]  # target prob for same token

        # Acceptance test: accept if draft wasn't too confident
        if q <= p or random.random() < p / (q + 1e-10):
            accepted.append(token_id)
        else:
            # Rejection: sample a corrected token from the adjusted distribution
            # Adjusted distribution = max(0, p−q) renormalized
            # This ensures final distribution = target distribution
            adjusted = np.maximum(0, target_probs[k] - draft_probs[k])
            adjusted_sum = adjusted.sum()
            if adjusted_sum > 1e-8:
                adjusted /= adjusted_sum
                bonus = int(np.random.choice(len(adjusted), p=adjusted))
            else:
                bonus = int(np.argmax(target_probs[k]))
            return accepted, bonus   # stop here, return rejected position

    # All K draft tokens accepted — sample bonus token from target at K+1
    bonus = int(np.random.choice(len(target_probs[K]), p=target_probs[K]))
    return accepted, bonus


# Simulate speculative decoding round
K = 4
vocab = 50

np.random.seed(99)
# Draft model generates K tokens with their probability distributions
draft_probs_raw = np.random.dirichlet(np.ones(vocab) * 0.5, size=K)
# Target model evaluates all K positions + 1 bonus position in one pass
target_probs_raw = np.random.dirichlet(np.ones(vocab) * 0.5, size=K+1)

draft_tokens_demo = [int(np.argmax(draft_probs_raw[k])) for k in range(K)]

accepted_toks, bonus_tok = speculative_decode_step(
    draft_probs_raw, target_probs_raw, draft_tokens_demo
)
print(f"\n[PART 15] Speculative Decoding:")
print(f"  Draft proposed {K} tokens: {draft_tokens_demo}")
print(f"  Accepted: {accepted_toks}  ({len(accepted_toks)}/{K} accepted)")
print(f"  Bonus token from target: {bonus_tok}")
print(f"  Total tokens this round: {len(accepted_toks) + 1} "
      f"(vs 1 without speculative decoding)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 16 ── PRODUCTION: QUANTIZATION, BATCHING, METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
QUANTIZATION — serving large models on smaller hardware:

  FP32 (full precision): 4 bytes/param  — training
  BF16 (bfloat16):       2 bytes/param  — modern training standard
  FP16 (float16):        2 bytes/param  — inference (narrower range than BF16)
  INT8:                  1 byte/param   — 2x compression vs FP16
  INT4 / NF4:            0.5 bytes/param — 4x compression vs FP16

  Llama 3 70B:
    BF16: 140GB → requires 2× A100 80GB
    INT8:  70GB → fits on 1× A100 80GB
    INT4:  35GB → fits on 1× RTX 4090!

  QUANTIZATION METHODS:
    RTN (Round-to-Nearest): fastest, naive, some quality loss
    GPTQ: post-training quantization using Hessian information. INT4 quality close to FP16.
    AWQ (Activation-aware): preserves important weight magnitudes. Best quality/speed.
    llama.cpp: CPU inference with GGUF format, k-quants (Q4_K_M etc.)

POST-TRAINING QUANTIZATION (PTQ) vs QUANTIZATION-AWARE TRAINING (QAT):
  PTQ: quantize after training. Fast, but quality drop.
  QAT: train with simulated quantization. Better quality, much more expensive.
"""

def absmax_quantize_int8(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Absmax INT8 quantization (simplest scheme).
    Maps FP32 weights to [-127, 127] integer range.
    scale factor stored for dequantization.
    """
    absmax = np.abs(weight).max()
    scale = 127.0 / (absmax + 1e-8)
    # Quantize: multiply by scale, round to nearest integer
    quantized = np.clip(np.round(weight * scale), -127, 127).astype(np.int8)
    return quantized, scale


def absmax_dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Recover FP32 approximation from INT8 + scale."""
    return quantized.astype(np.float32) / scale


"""
CONTINUOUS BATCHING (vLLM, TensorRT-LLM):
  Naive batching: wait for a full batch, run together, return all results.
  Problem: requests have different lengths. Padding wastes compute.
  Short requests finish early, GPU sits idle.

  Continuous batching: as soon as one request finishes, insert a new one.
  GPU is always processing K active requests. Dramatically higher throughput.
  Combined with paged attention for memory efficiency.

KEY PRODUCTION METRICS:

  TTFT (Time To First Token): latency from request → first generated token.
    Measures prefill latency. Target: < 200ms for good UX.

  TBT (Time Between Tokens): latency per generated token after first.
    Measures decode throughput. Target: < 50ms per token (20+ tok/s).

  Throughput: total tokens/second across all concurrent users.
    A100 with vLLM + Llama 3 70B INT4: ~1500 tokens/sec aggregate.

  MFU (Model FLOPs Utilization): actual FLOPs / theoretical peak FLOPs.
    Well-tuned training: 45-55% MFU. Poor setup: <30%.

  Memory bandwidth utilization (MBU): critical for inference.
    Inference is bandwidth-bound: measure GB/s achieved vs GPU peak.
"""

def simulate_continuous_batching(requests: list[dict]) -> list[dict]:
    """
    Conceptual continuous batching simulator.
    Shows how requests are processed as a dynamic batch.
    """
    results = []
    active = []   # currently running requests
    pending = list(requests)
    MAX_BATCH = 4

    step = 0
    while pending or active:
        # Fill batch up to MAX_BATCH
        while len(active) < MAX_BATCH and pending:
            req = pending.pop(0)
            req['tokens_generated'] = 0
            active.append(req)

        if not active:
            break

        # One decode step for all active requests
        finished = []
        for req in active:
            req['tokens_generated'] += 1
            if req['tokens_generated'] >= req['max_tokens']:
                finished.append(req)
                results.append({
                    'id': req['id'],
                    'tokens': req['tokens_generated'],
                    'finished_at_step': step
                })

        for req in finished:
            active.remove(req)

        step += 1

    return results


requests_demo = [
    {'id': 'r1', 'max_tokens': 3},
    {'id': 'r2', 'max_tokens': 5},
    {'id': 'r3', 'max_tokens': 2},
    {'id': 'r4', 'max_tokens': 4},
    {'id': 'r5', 'max_tokens': 6},
]

batch_results = simulate_continuous_batching(requests_demo)

print(f"\n[PART 16] Production — Quantization + Batching:")

# Quantization demo
W = np.array([[0.1, -0.5, 0.9, -1.2], [0.3, 0.7, -0.4, 0.8]], dtype=np.float32)
q8, scale = absmax_quantize_int8(W)
W_reconstructed = absmax_dequantize(q8, scale)
max_err = np.abs(W - W_reconstructed).max()

print(f"\n  INT8 Quantization:")
print(f"  Original   : {W.flatten().round(3)}")
print(f"  Quantized  : {q8.flatten()}")
print(f"  Scale      : {scale:.4f}")
print(f"  Reconstructed: {W_reconstructed.flatten().round(3)}")
print(f"  Max error  : {max_err:.6f}")
print(f"  Memory: {W.nbytes}B → {q8.nbytes}B ({100*q8.nbytes/W.nbytes:.0f}% of original)")

print(f"\n  Continuous Batching Results:")
for r in batch_results:
    print(f"  Request {r['id']}: {r['tokens']} tokens, finished at step {r['finished_at_step']}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE LLM PIPELINE SUMMARY                           ║
╠════════════════════════╦═══════════════════╦══════════════════════════════╣
║  COMPONENT             ║  KEY IDEA         ║  MODERN CHOICE               ║
╠════════════════════════╬═══════════════════╬══════════════════════════════╣
║  Tokenization          ║  Subword BPE      ║  SentencePiece / tiktoken    ║
║  Embedding             ║  Lookup table     ║  Weight-tied with LM head    ║
║  Positional encoding   ║  Inject position  ║  RoPE (rotary)               ║
║  Normalization         ║  Stable training  ║  RMSNorm (pre-norm)          ║
║  Attention             ║  Contextual repr  ║  GQA (grouped query)         ║
║  FFN activation        ║  Non-linearity    ║  SwiGLU                      ║
║  Residuals             ║  Gradient highway ║  Pre-Norm + Add              ║
║  Optimizer             ║  Adaptive LR      ║  AdamW + cosine schedule     ║
║  Fine-tuning           ║  Efficiency       ║  LoRA / QLoRA                ║
║  Alignment             ║  Human preference ║  DPO / RLHF / RLAIF         ║
║  Fast decoding         ║  Bandwidth-bound  ║  Speculative + KV cache      ║
║  Deployment            ║  Memory/speed     ║  INT4 quant + continuous bat ║
╚════════════════════════╩═══════════════════╩══════════════════════════════╝

WHAT TO STUDY NEXT:
  → Flash Attention 2/3 implementation (IO-aware tiling algorithm)
  → Mixture of Experts (Mixtral, GPT-4, DeepSeek) — sparse expert routing
  → Sliding Window Attention (Mistral) — for long contexts efficiently
  → Chain-of-thought / reasoning models (o1, DeepSeek-R1, GRPO)
  → Multimodal LLMs (vision encoders: LLaVA, Gemini, Claude 3)
  → Production deployment: vLLM, TensorRT-LLM, llama.cpp
""")
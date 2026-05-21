"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ADVANCED NLP WITH TRANSFORMERS — ANNOTATED REFERENCE CODE           ║
║                                                                              ║
║  TOPICS COVERED:                                                             ║
║  1.  Tokenization (WordPiece / BPE)                                         ║
║  2.  Embeddings (Token, Positional, Segment)                                ║
║  3.  Self-Attention & Multi-Head Attention                                  ║
║  4.  Transformer Encoder (BERT-style)                                       ║
║  5.  Transformer Decoder (GPT-style)                                        ║
║  6.  Pre-training Tasks (MLM, NSP, CLM)                                     ║
║  7.  Fine-tuning (Classification, NER, QA)                                  ║
║  8.  Sequence-to-Sequence (Translation / Summarization)                     ║
║  9.  Beam Search Decoding                                                   ║
║  10. Evaluation Metrics (BLEU, ROUGE, F1)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

WORKFLOW:
  Raw Text
      │
      ▼
  [1] Tokenization  →  token IDs + attention masks
      │
      ▼
  [2] Embeddings    →  dense vector representation (token + positional + segment)
      │
      ▼
  [3] Self-Attention →  context-aware token representations
      │
      ▼
  [4] Encoder Stack  →  deep contextual encodings (BERT / RoBERTa)
      │             or
  [5] Decoder Stack  →  auto-regressive generation (GPT / T5)
      │
      ▼
  [6] Task Head      →  classification / span extraction / generation
      │
      ▼
  [7] Fine-tuning    →  task-specific supervised learning
      │
      ▼
  [8] Decoding       →  greedy / beam search / nucleus sampling
      │
      ▼
  [9] Metrics        →  BLEU, ROUGE, F1, accuracy
"""

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIES
# pip install torch transformers datasets evaluate sacrebleu rouge_score
# ─────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,             # Unified tokenizer loader
    AutoModel,                 # Unified model loader
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    pipeline,                  # High-level inference API
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import evaluate
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOKENIZATION
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : Tokenization converts raw text into discrete integer IDs a model can
        process.  Modern transformers use sub-word schemes (WordPiece for BERT,
        Byte-Pair Encoding for GPT, SentencePiece for T5/mT5).

WHY   : Raw characters are too fine-grained (long sequences, huge vocab).
        Full words can't handle rare / unseen words.  Sub-words balance
        vocabulary size against sequence length and enable open-vocabulary NLP.

HOW   : The tokenizer has a fixed vocabulary (e.g., 30 522 tokens for BERT).
        It greedily splits words into the longest matching sub-word units,
        adding special tokens ([CLS], [SEP], <s>, </s>) required by the model.
"""

def demo_tokenization():
    # AutoTokenizer selects the correct tokenizer class from the model card
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Transformers revolutionized NLP by enabling parallelizable training."

    # --- Basic encoding ---
    # return_tensors="pt"  → PyTorch tensors (use "tf" for TensorFlow)
    # padding=True         → pad shorter sequences in a batch to the same length
    # truncation=True      → cut sequences longer than max_length tokens
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    print("Token IDs  :", encoding["input_ids"])
    # attention_mask: 1 for real tokens, 0 for padding — prevents attending to pad
    print("Attn Mask  :", encoding["attention_mask"])
    # Decode back to human-readable tokens (including special tokens)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    print("Tokens     :", tokens)

    # --- Batch encoding (multiple sentences) ---
    batch = tokenizer(
        ["Hello world!", "Attention is all you need."],
        return_tensors="pt",
        padding=True,      # pads the shorter sentence to match the longer
        truncation=True,
    )
    print("\nBatch input_ids shape:", batch["input_ids"].shape)
    # Output: [2, max_seq_len] — (batch_size, sequence_length)

    # --- Token type IDs (segment IDs) — used in BERT NSP task ---
    # Sentence A tokens → 0, Sentence B tokens → 1
    pair_encoding = tokenizer(
        "What is NLP?",
        "Natural Language Processing is a field of AI.",
        return_tensors="pt",
    )
    print("Token type IDs:", pair_encoding["token_type_ids"])

    return tokenizer, encoding


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EMBEDDINGS (from scratch, for intuition)
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : An embedding maps a discrete token ID to a continuous dense vector.
        Transformers use THREE additive embedding components:
          a) Token Embedding    — learned meaning of the token itself
          b) Positional Encoding — injects order information (sin/cos or learned)
          c) Segment Embedding  — differentiates sentence A from sentence B

WHY   : Neural networks need floating-point inputs.  Positional encoding is
        critical because self-attention is PERMUTATION INVARIANT — without it,
        "dog bites man" and "man bites dog" look identical.

HOW   : The three vectors are SUMMED, then passed through LayerNorm + Dropout
        to produce the initial hidden state for the transformer stack.
"""

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position: int = 512, num_segments: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        # (a) Token embedding: maps vocab index → hidden_size vector
        #     This is a simple lookup table (nn.Embedding is just a matrix).
        self.token_embed = nn.Embedding(vocab_size, hidden_size,
                                        padding_idx=0)  # pad token gets zero grad

        # (b) Positional embedding (learned, as in BERT)
        #     Alternative: fixed sinusoidal (as in original "Attention is All You Need")
        self.position_embed = nn.Embedding(max_position, hidden_size)

        # (c) Segment embedding: helps BERT distinguish two sentences in a pair
        self.segment_embed = nn.Embedding(num_segments, hidden_size)

        # LayerNorm stabilizes training by normalizing activations per sample.
        # eps prevents division by zero for near-zero variance inputs.
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Dropout randomly zeroes out neurons during training → regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)

        # Build position tensor [0, 1, 2, ..., seq_len-1] for each sample
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Sum all three embedding contributions
        embeddings = (
            self.token_embed(input_ids)       # semantic content
            + self.position_embed(positions)  # positional order
            + self.segment_embed(token_type_ids)  # sentence identity
        )

        return self.dropout(self.layer_norm(embeddings))


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) sinusoidal encoding from the original Transformer paper.
    Uses sine for even dimensions, cosine for odd dimensions.

    WHY sinusoidal? The model can extrapolate to longer sequences at inference
    than it saw during training, because the patterns are mathematically defined.
    """
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # Scaling factor: 10000^(2i/d_model) — creates different frequencies
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims → cos
        # Register as buffer (not a parameter, but moves with .to(device))
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, d]

    def forward(self, x):
        # Add positional encoding to token embeddings
        return x + self.pe[:, :x.size(1)]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SELF-ATTENTION & MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : Self-attention lets every token attend to every other token in the
        sequence, computing a weighted average of all value vectors.

WHY   : Captures long-range dependencies that RNNs struggled with (gradient
        vanishing).  Parallelizable across positions (unlike sequential RNNs).

HOW   : For each token, compute three vectors via learned linear projections:
          Q (Query)  — "what am I looking for?"
          K (Key)    — "what do I contain?"
          V (Value)  — "what information do I carry?"
        Attention score = softmax(Q·Kᵀ / √d_k) · V
        The √d_k scaling prevents the dot products from growing too large
        in high dimensions, which would saturate softmax → near-zero gradients.

        Multi-Head Attention runs H independent attention heads in parallel,
        each in a lower-dimensional subspace (d_k = d_model / H).  Different
        heads can specialize: syntax, coreference, semantics, etc.
        Outputs are concatenated and projected back to d_model.
"""

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V shapes: [batch, heads, seq_len, d_k]
        mask         : [batch, 1, 1, seq_len]  — True where positions are MASKED
        """
        d_k = Q.size(-1)  # dimension of each head

        # Raw attention scores — dot product similarity between queries and keys
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # Shape: [batch, heads, seq_len_q, seq_len_k]

        if mask is not None:
            # Fill masked positions with -∞ so softmax assigns them ~0 weight
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax converts scores to a probability distribution over positions
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of value vectors
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads  # dimension per head

        # Linear projections for Q, K, V (all heads computed together)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        # Output projection to merge all heads back to hidden_size
        self.W_o = nn.Linear(hidden_size, hidden_size)

        self.attention = ScaledDotProductAttention(dropout)

    def split_heads(self, x, batch_size):
        """Reshape [batch, seq, hidden] → [batch, heads, seq, d_k]"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # swap seq and heads dims

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and split into multiple heads
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key),   batch_size)
        V = self.split_heads(self.W_v(value),  batch_size)

        # Run attention in each head's subspace
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate all heads: [batch, heads, seq, d_k] → [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.num_heads * self.d_k)

        # Final linear projection
        return self.W_o(attn_output), attn_weights


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRANSFORMER ENCODER BLOCK (BERT-style)
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : A stack of identical encoder layers.  Each layer has:
          1. Multi-Head Self-Attention (all tokens attend to all tokens)
          2. Position-wise Feed-Forward Network (two linear layers + GELU/ReLU)
        Both sublayers are wrapped in Add & LayerNorm (residual connections).

WHY   : Residual connections allow gradients to flow directly through layers
        (prevents vanishing gradients in deep networks).  The FFN adds
        non-linearity and per-position feature transformation capacity.

HOW   : BERT-base = 12 encoder layers, 12 heads, hidden_size=768, 110M params.
        Input: [CLS] token1 token2 ... [SEP] — bidirectional context.
        [CLS] final hidden state → classification tasks.
"""

class FeedForward(nn.Module):
    """
    Position-wise FFN: expands to 4× hidden_size with GELU, then projects back.
    GELU (Gaussian Error Linear Unit) outperforms ReLU on NLP tasks.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn       = FeedForward(hidden_size, dropout)
        # Two separate LayerNorms — one after attention, one after FFN
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Sublayer 1: Self-Attention with residual connection ---
        attn_output, _ = self.self_attn(x, x, x, mask)  # query=key=value=x (self-attention)
        x = self.norm1(x + self.dropout(attn_output))    # Add & Norm

        # --- Sublayer 2: Feed-Forward with residual connection ---
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))     # Add & Norm

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12,
                 num_heads=12, max_position=512, dropout=0.1):
        super().__init__()
        self.embeddings = TransformerEmbeddings(vocab_size, hidden_size,
                                                max_position, dropout=dropout)
        # Stack of N identical encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # Reshape mask for broadcasting: [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)  # [batch, seq_len, hidden_size]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRANSFORMER DECODER (GPT / Causal LM style)
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : The decoder is used in auto-regressive language models (GPT) and the
        decoder side of seq2seq models (T5, BART).  It adds:
          1. MASKED Self-Attention   — token i can only attend to tokens 0..i
          2. Cross-Attention         — queries from decoder, K/V from encoder
          3. FFN (same as encoder)

WHY   : Masking the future positions enforces the auto-regressive property:
        the model predicts the next token using only past context.  This is
        implemented with a causal (triangular) mask.

HOW   : The causal mask is an upper-triangular matrix of -∞ values, zeroing
        out all future positions in the softmax distribution.
"""

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Creates a lower-triangular boolean mask.
    Position (i, j) = True  if j <= i  (token j is visible from position i)
                    = False if j >  i  (future token — masked out)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # 1. Masked self-attention (causal — for auto-regressive generation)
        self.masked_self_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        # 2. Cross-attention (only present in encoder-decoder architectures)
        self.cross_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        # 3. Feed-forward
        self.ffn = FeedForward(hidden_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, src_mask=None, tgt_mask=None):
        # --- Sublayer 1: Masked self-attention (causal) ---
        attn1, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # --- Sublayer 2: Cross-attention (skip if decoder-only like GPT) ---
        if encoder_output is not None:
            # Query from decoder, Key/Value from encoder output
            attn2, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
            x = self.norm2(x + self.dropout(attn2))

        # --- Sublayer 3: Feed-forward ---
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PRE-TRAINING TASKS
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : Transformers learn general language representations through self-
        supervised pre-training on massive unlabeled corpora.

  MLM (Masked Language Model) — BERT's pre-training:
    - Randomly mask 15% of tokens; model predicts the originals.
    - 80% → [MASK], 10% → random token, 10% → unchanged.
    - Enables bidirectional context learning.

  NSP (Next Sentence Prediction) — BERT:
    - Predict whether sentence B follows sentence A in original document.
    - Later shown to be less useful; RoBERTa dropped it.

  CLM (Causal Language Model) — GPT's pre-training:
    - Predict the next token given all previous tokens (left-to-right).
    - Naturally suited for text generation tasks.

WHY   : Pre-training on billions of tokens gives the model deep syntactic,
        semantic, and world-knowledge representations.  Fine-tuning on a
        small task-specific dataset then adapts these representations cheaply.
"""

def mask_tokens_for_mlm(input_ids: torch.Tensor, tokenizer, mlm_probability=0.15):
    """
    Implements BERT's MLM masking strategy.
    Returns (masked_input_ids, labels) where labels=-100 means 'ignore in loss'.
    """
    labels = input_ids.clone()
    # Probability matrix — True where we will apply the masking strategy
    prob_matrix = torch.full(labels.shape, mlm_probability)

    # Don't mask special tokens ([CLS], [SEP], [PAD])
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        special_tokens_mask |= (labels == special_id)
    prob_matrix[special_tokens_mask] = 0.0

    masked_indices = torch.bernoulli(prob_matrix).bool()  # positions to process
    labels[~masked_indices] = -100  # only compute loss on masked positions

    # 80% of masked → replace with [MASK]
    replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[replace_with_mask] = tokenizer.mask_token_id

    # 10% of masked → replace with random token
    replace_with_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices & ~replace_with_mask
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[replace_with_random] = random_words[replace_with_random]

    # Remaining 10% → keep original (model sees the real token but still predicts it)
    return input_ids, labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FINE-TUNING WITH HuggingFace Trainer
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : Fine-tuning adapts a pre-trained model to a downstream NLP task by
        training a small task-specific "head" (+ updating encoder weights)
        on labeled data.

Tasks:
  (a) Sequence Classification — sentiment, topic, entailment
  (b) Token Classification    — Named Entity Recognition (NER), POS tagging
  (c) Question Answering      — extractive span prediction

WHY   : Pre-trained representations need only small data and few epochs to
        reach high performance on downstream tasks.  Full pre-training would
        require weeks of GPU compute; fine-tuning takes minutes to hours.

HOW   : The Trainer API handles training loops, evaluation, gradient
        accumulation, mixed-precision, checkpointing, and logging.
"""

# ── (a) Sequence Classification (Sentiment Analysis) ─────────────────────────

def fine_tune_sentiment():
    """Fine-tune BERT on SST-2 binary sentiment classification."""

    model_name = "bert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    # AutoModelForSequenceClassification adds a linear layer on top of [CLS]:
    #   hidden_size → num_labels   (+ softmax at inference)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2  # positive / negative
    )

    dataset = load_dataset("glue", "sst2")

    def preprocess(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=128,
        )

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")

    # DataCollatorWithPadding pads batches to the longest sequence dynamically
    # (more efficient than padding to global max_length)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Evaluation metric
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir="./results/sentiment",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,           # typical fine-tuning LR (smaller than pre-training)
        warmup_ratio=0.06,            # linear warmup for first 6% of steps
        weight_decay=0.01,            # L2 regularization
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # mixed-precision for faster training
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"Validation accuracy: {results['eval_accuracy']:.4f}")
    return trainer


# ── (b) Token Classification (Named Entity Recognition) ──────────────────────

def fine_tune_ner():
    """Fine-tune BERT on CoNLL-2003 NER (PER, ORG, LOC, MISC)."""

    model_name = "bert-base-cased"   # Cased model for NER (capitalization matters)
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    dataset  = load_dataset("conll2003")
    label_names = dataset["train"].features["ner_tags"].feature.names
    # e.g. ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', ...]

    def tokenize_and_align_labels(batch):
        """
        Sub-word tokenization splits words into multiple tokens.
        We must align the word-level NER labels to sub-word tokens:
          - First sub-word of a word → original label
          - Subsequent sub-words     → -100 (ignored in loss)
        """
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,  # input is pre-tokenized (list of words)
        )
        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)  # maps token → word index
            aligned  = []
            prev_word = None
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)           # special token → ignore
                elif word_id != prev_word:
                    aligned.append(labels[word_id])  # first sub-token of word
                else:
                    aligned.append(-100)           # continuation sub-token → ignore
                prev_word = word_id
            all_labels.append(aligned)
        tokenized["labels"] = all_labels
        return tokenized

    tokenized = dataset.map(
        tokenize_and_align_labels, batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # AutoModelForTokenClassification adds a linear layer per token:
    #   [batch, seq_len, hidden_size] → [batch, seq_len, num_labels]
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_names)
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        # Convert IDs back to label strings, excluding -100 (ignored) positions
        true_preds  = [[label_names[p] for p, l in zip(pred, label) if l != -100]
                       for pred, label in zip(preds, labels)]
        true_labels = [[label_names[l] for p, l in zip(pred, label) if l != -100]
                       for pred, label in zip(preds, labels)]
        result = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": result["overall_precision"],
            "recall":    result["overall_recall"],
            "f1":        result["overall_f1"],
        }

    training_args = TrainingArguments(
        output_dir="./results/ner",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


# ── (c) Extractive Question Answering ────────────────────────────────────────

def fine_tune_qa():
    """
    Fine-tune BERT on SQuAD v1.1.
    The model predicts START and END token positions within the passage.
    AutoModelForQuestionAnswering adds two linear layers:
      hidden_size → 1  (start logits)
      hidden_size → 1  (end logits)
    """
    model_name = "bert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Full SQuAD preprocessing is verbose; here we show the key encoding step
    def preprocess_qa(batch):
        # Encode question + context as a pair
        inputs = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",  # truncate the context (not the question)
            max_length=384,
            stride=128,               # overlap between chunks for long contexts
            return_overflowing_tokens=True,  # split long contexts into multiple spans
            return_offsets_mapping=True,
            padding="max_length",
        )
        return inputs

    print("QA model ready for fine-tuning on SQuAD.")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SEQUENCE-TO-SEQUENCE (Translation / Summarization)
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : Seq2Seq models (T5, BART, mBART) have a full encoder-decoder architecture.
        The encoder reads the source; the decoder generates the target token by token.

WHY   : Generation tasks can't be solved with classification heads — the output
        is a variable-length sequence from a large vocabulary.

HOW   : Teacher forcing during training: at each decoder step, feed the GROUND-
        TRUTH previous token (not the model's prediction) for stable training.
        At inference, use the model's own previous prediction (auto-regressive).
"""

def fine_tune_summarization():
    """Fine-tune T5-small on CNN/DailyMail summarization."""

    model_name = "t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_dataset("cnn_dailymail", "3.0.0")

    max_input_length  = 512
    max_target_length = 128

    def preprocess(batch):
        # T5 uses a text-to-text format: prefix describes the task
        inputs  = ["summarize: " + doc for doc in batch["article"]]
        targets = batch["highlights"]

        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )
        # Tokenize the target summaries as labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=True
            )
        # Replace padding token id in labels with -100 → ignored in cross-entropy loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True,
                            remove_columns=dataset["train"].column_names)

    # DataCollatorForSeq2Seq pads both encoder input and decoder labels
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits[0], axis=-1)
        # Decode predictions and references, skipping -100 labels
        decoded_preds   = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels          = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels  = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v, 4) for k, v in result.items()}

    training_args = TrainingArguments(
        output_dir="./results/summarization",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        predict_with_generate=True,  # use model.generate() during eval
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized["train"].select(range(10000)),  # subset for demo
        eval_dataset=tokenized["validation"].select(range(500)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — DECODING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : At inference, the model outputs a probability distribution over the
        vocabulary at each step.  Decoding strategies determine which token
        to pick next.

  Greedy     : Always pick the highest-probability token.  Fast but repetitive.
  Beam Search: Maintain top-K partial hypotheses.  Better quality, more compute.
  Top-K      : Sample from the K highest-probability tokens (creativity vs quality).
  Top-P      : Sample from the smallest set of tokens whose cumulative prob ≥ p
               (nucleus sampling — adapts vocabulary dynamically).
  Temperature: Divide logits by T before softmax.  T<1 sharpens, T>1 flattens.

WHY   : No single strategy is best for all tasks.  Greedy/beam for translation;
        nucleus sampling for creative text; low temperature for factual tasks.
"""

def demonstrate_decoding():
    model_name = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=model_name,
                          device=0 if torch.cuda.is_available() else -1)

    article = """
    The transformer architecture, introduced in the 2017 paper "Attention Is All You Need"
    by Vaswani et al., replaced recurrent networks with self-attention mechanisms.
    This change enabled massively parallel training and led to models like BERT and GPT
    that achieved state-of-the-art results across virtually every NLP benchmark.
    """

    # --- Greedy decoding ---
    greedy_out = summarizer(article, max_length=60, min_length=20,
                            num_beams=1, do_sample=False)

    # --- Beam search (4 beams) ---
    beam_out   = summarizer(article, max_length=60, min_length=20,
                            num_beams=4, early_stopping=True)

    # --- Nucleus sampling ---
    sample_out = summarizer(article, max_length=60, min_length=20,
                            do_sample=True, top_p=0.92, temperature=0.8)

    print("Greedy  :", greedy_out[0]["summary_text"])
    print("Beam-4  :", beam_out[0]["summary_text"])
    print("Nucleus :", sample_out[0]["summary_text"])

    return greedy_out, beam_out, sample_out


def beam_search_from_scratch(model, tokenizer, prompt: str,
                             num_beams: int = 4, max_new_tokens: int = 50):
    """
    Minimal beam search implementation to illustrate the algorithm.
    In practice, use model.generate(num_beams=N) which is far more optimized.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Each beam: (cumulative_log_prob, token_ids)
    beams = [(0.0, input_ids)]
    completed = []

    for _ in range(max_new_tokens):
        all_candidates = []
        for score, ids in beams:
            with torch.no_grad():
                output = model(ids)
                next_logits = output.logits[:, -1, :]  # logits for next token
                next_probs  = F.log_softmax(next_logits, dim=-1)

            # Expand each beam by top-K tokens
            top_scores, top_ids = next_probs.topk(num_beams)
            for beam_score, beam_token in zip(top_scores[0], top_ids[0]):
                new_ids   = torch.cat([ids, beam_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                new_score = score + beam_score.item()
                all_candidates.append((new_score, new_ids))

        # Keep top num_beams candidates by score (length-normalized for fairness)
        all_candidates.sort(key=lambda x: x[0] / x[1].size(-1), reverse=True)
        beams = all_candidates[:num_beams]

        # Check for EOS token
        eos_id = tokenizer.eos_token_id
        if eos_id and any(ids[0, -1].item() == eos_id for _, ids in beams):
            break

    best_ids = beams[0][1]
    return tokenizer.decode(best_ids[0], skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : Quantitative measures of model output quality.

  BLEU  (Bilingual Evaluation Understudy):
    - Precision-based n-gram overlap between prediction and reference.
    - Penalizes short outputs via a brevity penalty.
    - Standard for machine translation.

  ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
    - ROUGE-N: n-gram overlap (both precision and recall).
    - ROUGE-L: longest common subsequence.
    - Standard for summarization.

  F1 Score (for NER / QA):
    - Harmonic mean of precision and recall at the entity / span level.

WHY   : Automated metrics enable cheap, reproducible evaluation at scale.
        They correlate imperfectly with human judgment but are essential for
        model selection and hyperparameter tuning.
"""

def evaluate_metrics(predictions: list, references: list, task: str = "translation"):
    if task == "translation":
        bleu = evaluate.load("sacrebleu")
        # SacreBLEU expects references as list of lists
        result = bleu.compute(predictions=predictions,
                              references=[[r] for r in references])
        print(f"BLEU score: {result['score']:.2f}")
        return result

    elif task == "summarization":
        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=predictions, references=references)
        for k, v in result.items():
            print(f"{k.upper()}: {v:.4f}")
        return result

    elif task == "ner":
        seqeval = evaluate.load("seqeval")
        result  = seqeval.compute(predictions=predictions, references=references)
        print(f"F1   : {result['overall_f1']:.4f}")
        print(f"Prec : {result['overall_precision']:.4f}")
        print(f"Rec  : {result['overall_recall']:.4f}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — INFERENCE WITH HIGH-LEVEL PIPELINES
# ══════════════════════════════════════════════════════════════════════════════
"""
WHAT  : HuggingFace's pipeline() provides a one-line API for common NLP tasks.
WHY   : Wraps tokenization, model forward pass, and post-processing into one call.
HOW   : Internally calls tokenizer → model → argmax / generate → decode.
"""

def run_all_pipelines():
    sample_text = (
        "Apple Inc. was founded by Steve Jobs in Cupertino, California. "
        "The company is worth over two trillion dollars."
    )

    # --- Sentiment Analysis ---
    sentiment = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Sentiment:", sentiment(sample_text))

    # --- Named Entity Recognition ---
    ner = pipeline("ner", model="dslim/bert-base-NER",
                   aggregation_strategy="simple")
    print("NER:", ner(sample_text))

    # --- Summarization ---
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("Summary:", summarizer(sample_text, max_length=40, min_length=10))

    # --- Zero-Shot Classification (no fine-tuning needed!) ---
    # Uses NLI (Natural Language Inference) internally:
    # "this text is about {label}" → entailment score → label probability
    zero_shot = pipeline("zero-shot-classification",
                         model="facebook/bart-large-mnli")
    print("Zero-shot:", zero_shot(
        sample_text,
        candidate_labels=["technology", "sports", "finance", "politics"]
    ))

    # --- Text Generation (GPT-2) ---
    generator = pipeline("text-generation", model="gpt2")
    print("Generated:", generator("The future of NLP is",
                                  max_new_tokens=30, num_return_sequences=1))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — RUN DEMOS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 1 — TOKENIZATION")
    print("=" * 70)
    tokenizer, encoding = demo_tokenization()

    print("\n" + "=" * 70)
    print("SECTION 2 — EMBEDDINGS (custom module instantiation)")
    print("=" * 70)
    embed_module = TransformerEmbeddings(vocab_size=30522, hidden_size=768)
    dummy_ids    = torch.randint(0, 30522, (2, 16))  # batch=2, seq=16
    emb_output   = embed_module(dummy_ids)
    print(f"Embedding output shape: {emb_output.shape}")  # [2, 16, 768]

    print("\n" + "=" * 70)
    print("SECTION 3 — MULTI-HEAD ATTENTION")
    print("=" * 70)
    mha    = MultiHeadAttention(hidden_size=768, num_heads=12)
    x_in   = torch.randn(2, 16, 768)
    attn_out, weights = mha(x_in, x_in, x_in)
    print(f"Attention output shape : {attn_out.shape}")   # [2, 16, 768]
    print(f"Attention weights shape: {weights.shape}")    # [2, 12, 16, 16]

    print("\n" + "=" * 70)
    print("SECTION 4 — ENCODER STACK")
    print("=" * 70)
    encoder = TransformerEncoder(vocab_size=30522, hidden_size=256,
                                  num_layers=2, num_heads=8)
    ids     = torch.randint(0, 30522, (2, 16))
    mask    = torch.ones(2, 16, dtype=torch.long)
    enc_out = encoder(ids, attention_mask=mask)
    print(f"Encoder output shape: {enc_out.shape}")  # [2, 16, 256]

    print("\n" + "=" * 70)
    print("SECTION 9 — PIPELINE INFERENCE")
    print("=" * 70)
    run_all_pipelines()

    print("\nAll demos complete.")
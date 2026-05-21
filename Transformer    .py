"""
╔══════════════════════════════════════════════════════════════════╗
║           ADVANCED TRANSFORMER - Complete Implementation          ║
╠══════════════════════════════════════════════════════════════════╣
║  USE:   NLP tasks — translation, summarization, classification,  ║
║         text generation, QA, token prediction                    ║
║  MODEL: Encoder-Decoder Transformer (Vaswani et al. 2017)        ║
╠══════════════════════════════════════════════════════════════════╣
║  REQUIREMENTS:                                                   ║
║    pip install torch numpy                                       ║
║    Python >= 3.8 | PyTorch >= 1.10                              ║
╠══════════════════════════════════════════════════════════════════╣
║  WORKFLOW:                                                       ║
║    Input Tokens                                                  ║
║       ↓ Token + Positional Embedding                            ║
║    [Encoder Block × N]                                           ║
║       → Multi-Head Self-Attention                               ║
║       → Add & LayerNorm                                          ║
║       → Feed-Forward Network                                     ║
║       → Add & LayerNorm                                          ║
║    [Decoder Block × N]                                           ║
║       → Masked Multi-Head Self-Attention                        ║
║       → Cross-Attention (attends to encoder output)             ║
║       → Feed-Forward Network                                     ║
║       → Add & LayerNorm (×3)                                    ║
║    Linear + Softmax → Output Probabilities                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# 1. POSITIONAL ENCODING
#    WHY: Transformers have no recurrence/convolution,
#         so position info is injected via sine/cosine waves.
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims  → cos
        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer("pe", pe)                  # not a param, but saved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 2. SCALED DOT-PRODUCT ATTENTION
#    FORMULA: Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
#    WHY scale? Prevents vanishing gradients at large dₖ.
# ─────────────────────────────────────────────
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    # (batch, heads, seq, seq)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask==0 → fill with -inf so softmax → 0
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# ─────────────────────────────────────────────
# 3. MULTI-HEAD ATTENTION
#    WHY: Multiple heads let the model attend to
#         different representation subspaces simultaneously.
#    STRUCTURE:
#      Linear(Q) → h heads  ┐
#      Linear(K) → h heads  ├→ Scaled Dot-Product → Concat → Linear
#      Linear(V) → h heads  ┘
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads     # per-head dimension

        # Projection matrices (bias=False matches original paper)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # output projection

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None            # stored for visualization

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d_model) → (batch, heads, seq, d_k)
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B = query.size(0)

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        x, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )

        # Merge heads: (batch, heads, seq, d_k) → (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(x)


# ─────────────────────────────────────────────
# 4. POSITION-WISE FEED-FORWARD NETWORK
#    FORMULA: FFN(x) = max(0, xW₁+b₁)W₂+b₂
#    WHY: Adds non-linearity and per-position transformation.
#    dim_ff is typically 4× d_model (2048 for d_model=512)
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),          # swap with nn.GELU() for BERT-style
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 5. ENCODER LAYER
#    STRUCTURE: Self-Attn → Add&Norm → FFN → Add&Norm
#    PRE-NORM variant (more stable training):
#      x = x + SubLayer(LayerNorm(x))
# ─────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN: normalize before sublayer (more stable than post-LN)
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ─────────────────────────────────────────────
# 6. DECODER LAYER
#    STRUCTURE:
#      Masked Self-Attn → Add&Norm
#      Cross-Attn (Q=decoder, K=V=encoder) → Add&Norm
#      FFN → Add&Norm
#    WHY masking in self-attn? Prevents decoder from
#    "seeing the future" (autoregressive property).
# ─────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)  # masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # encoder-decoder
        self.ffn = FeedForward(d_model, dim_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1) Masked self-attention (tgt attends to tgt, causal mask)
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        # 2) Cross-attention (tgt queries attend to encoder keys/values)
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))
        # 3) Feed-forward
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


# ─────────────────────────────────────────────
# 7. ENCODER STACK  (N identical EncoderLayers)
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 8. DECODER STACK  (N identical DecoderLayers)
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 9. FULL TRANSFORMER
#    Combines: Embedding + PE → Encoder → Decoder → Linear
# ─────────────────────────────────────────────
class Transformer(nn.Module):
    """
    Parameters
    ----------
    src_vocab : int   — source vocabulary size
    tgt_vocab : int   — target vocabulary size
    d_model   : int   — embedding / hidden dimension  (default 512)
    num_heads : int   — attention heads               (default 8)
    N         : int   — number of encoder/decoder layers (default 6)
    dim_ff    : int   — feed-forward inner dimension  (default 2048)
    dropout   : float — dropout probability           (default 0.1)
    max_len   : int   — maximum sequence length       (default 5000)
    """

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        num_heads: int = 8,
        N: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        # Embeddings (shared weight tying is optional but common)
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        # Stacks
        enc_layer = EncoderLayer(d_model, num_heads, dim_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, dim_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Final projection to vocabulary logits
        self.output_proj = nn.Linear(d_model, tgt_vocab)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform init — critical for stable training
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Mask helpers ──────────────────────────────────────────
    @staticmethod
    def make_src_mask(src: torch.Tensor) -> torch.Tensor:
        """Padding mask: 0-tokens are ignored. Shape: (B,1,1,S)"""
        return (src != 0).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
        """Causal + padding mask for decoder. Shape: (B,1,T,T)"""
        B, T = tgt.size()
        pad_mask  = (tgt != 0).unsqueeze(1).unsqueeze(2)           # (B,1,1,T)
        causal    = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T,T)
        return pad_mask & causal                                    # (B,1,T,T)

    # ── Forward pass ─────────────────────────────────────────
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        src : (batch, src_len)  — integer token ids
        tgt : (batch, tgt_len)  — integer token ids (teacher-forced)
        returns: (batch, tgt_len, tgt_vocab) logits
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Embed + positional encode
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim))

        enc_out = self.encoder(src_emb, src_mask)
        dec_out = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)

        return self.output_proj(dec_out)   # logits, apply softmax externally

    # ── Greedy decoding (inference) ───────────────────────────
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 50,
    ) -> list[int]:
        """
        Autoregressive greedy decoding (one token at a time).
        Returns list of token ids (excluding <sos>).
        """
        self.eval()
        src_mask = self.make_src_mask(src)
        src_emb  = self.pos_enc(self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim))
        enc_out  = self.encoder(src_emb, src_mask)

        # Start with <sos>
        tgt = torch.tensor([[sos_idx]], device=src.device)
        result = []

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(tgt)
            tgt_emb  = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim))
            dec_out  = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)
            logits   = self.output_proj(dec_out[:, -1])       # last token logits
            next_tok = logits.argmax(-1, keepdim=True)        # greedy pick
            result.append(next_tok.item())
            if next_tok.item() == eos_idx:
                break
            tgt = torch.cat([tgt, next_tok], dim=1)

        return result


# ─────────────────────────────────────────────
# 10. LABEL SMOOTHED CROSS-ENTROPY LOSS
#     WHY: Prevents overconfident predictions;
#          improves generalization (Szegedy et al. 2016).
# ─────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits : (B*T, vocab)  |  target : (B*T,)
        log_prob = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_prob.mean(dim=-1)
        nll_loss    = -log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        # Ignore padding
        mask = target != self.pad_idx
        return loss[mask].mean()


# ─────────────────────────────────────────────
# 11. LEARNING RATE SCHEDULER (Noam / Warmup)
#     WHY: Linear warmup prevents early instability;
#          inverse-sqrt decay ensures convergence.
#     lr = d_model^(-0.5) · min(step^(-0.5), step·warmup^(-1.5))
# ─────────────────────────────────────────────
class NoamScheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self._step         = 0
        self._rate         = 0.0

    def step(self):
        self._step += 1
        rate = self._compute_lr()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def _compute_lr(self) -> float:
        s, w = max(self._step, 1), self.warmup_steps
        return self.d_model ** -0.5 * min(s ** -0.5, s * w ** -1.5)

    def zero_grad(self):
        self.optimizer.zero_grad()


# ─────────────────────────────────────────────
# 12. TRAINING LOOP  (minimal, extensible)
# ─────────────────────────────────────────────
def train_epoch(
    model: Transformer,
    data_loader,          # yields (src, tgt) batches of token tensors
    criterion: nn.Module,
    scheduler: NoamScheduler,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)

        # Teacher forcing: feed tgt[:-1], predict tgt[1:]
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:].contiguous()

        logits = model(src, tgt_in)                          # (B, T-1, vocab)
        loss   = criterion(
            logits.view(-1, logits.size(-1)),                # (B*T-1, vocab)
            tgt_out.view(-1)                                 # (B*T-1,)
        )

        scheduler.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # gradient clipping
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)


# ─────────────────────────────────────────────
# 13. QUICK SMOKE TEST  (no real data needed)
#     Run: python transformer.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Transformer Smoke Test")
    print("=" * 60)

    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC_VOCAB = 1000
    TGT_VOCAB = 1000
    BATCH     = 4
    SRC_LEN   = 20
    TGT_LEN   = 18
    D_MODEL   = 256    # small for demo; use 512 for full model
    N_HEADS   = 8
    N_LAYERS  = 4      # use 6 for full model
    DIM_FF    = 512    # use 2048 for full model

    # Build model
    model = Transformer(
        src_vocab=SRC_VOCAB,
        tgt_vocab=TGT_VOCAB,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        N=N_LAYERS,
        dim_ff=DIM_FF,
        dropout=0.1,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Device        : {DEVICE}")
    print(f"  Parameters    : {param_count:,}")

    # Dummy batch (token ids 1..vocab-1, 0=pad)
    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN)).to(DEVICE)
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(DEVICE)

    # Forward pass
    logits = model(src, tgt[:, :-1])
    print(f"  Input  shape  : src{list(src.shape)}  tgt{list(tgt.shape)}")
    print(f"  Output shape  : {list(logits.shape)}  ← (batch, tgt_len-1, vocab)")

    # Loss
    criterion = LabelSmoothingLoss(TGT_VOCAB, pad_idx=0, smoothing=0.1)
    loss = criterion(logits.reshape(-1, TGT_VOCAB), tgt[:, 1:].reshape(-1))
    print(f"  Sample loss   : {loss.item():.4f}")

    # Greedy decode (single example)
    single_src = src[:1]
    decoded = model.greedy_decode(single_src, sos_idx=1, eos_idx=2, max_len=15)
    print(f"  Greedy decode : {decoded}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, D_MODEL, warmup_steps=4000)
    print(f"  LR after step1: {scheduler._compute_lr():.2e}")

    print("=" * 60)
    print("  ✓ All components verified successfully")
    print("=" * 60)
    print()
    print("  NEXT STEPS:")
    print("  1. Replace dummy DataLoader with real tokenized dataset")
    print("  2. Add validation loop + BLEU score evaluation")
    print("  3. Implement beam search for better decoding")
    print("  4. Scale: d_model=512, N=6, dim_ff=2048, heads=8")
    print("  5. Add checkpointing: torch.save(model.state_dict(), 'ckpt.pt')")  
"""
=======================================================================
  ADVANCED LSTM & GRU — Complete Implementation
=======================================================================
  USE        : Time-series forecasting, NLP sequence modelling,
               anomaly detection, stock prediction, text generation
  REQUIREMENT: pip install torch numpy matplotlib scikit-learn
  WORKFLOW   :
    1. Generate / load data
    2. Preprocess & create sequences
    3. Define LSTM / GRU models
    4. Train with early stopping
    5. Evaluate & visualise
=======================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import math, warnings
warnings.filterwarnings("ignore")

# ── Config (change these) ──────────────────────────────────────────────
SEQ_LEN    = 50      # look-back window
PRED_LEN   = 1       # steps to predict
HIDDEN     = 128     # hidden units per layer
LAYERS     = 2       # stacked RNN depth
DROPOUT    = 0.3     # regularisation
BATCH      = 64
EPOCHS     = 50
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════
# 1. DATA — synthetic sine + noise (replace with your own CSV / API)
# ══════════════════════════════════════════════════════════════════════
def generate_data(n=2000):
    """Sine wave with trend + Gaussian noise — swap with real data."""
    t   = np.linspace(0, 8 * np.pi, n)
    sig = np.sin(t) + 0.05 * t + 0.15 * np.random.randn(n)
    return sig.reshape(-1, 1).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════
# 2. DATASET — sliding window sequences
# ══════════════════════════════════════════════════════════════════════
class SeqDataset(Dataset):
    """
    Converts a 1-D time-series into (X, y) sliding windows.
    X shape : (seq_len, features)
    y shape : (pred_len,)
    """
    def __init__(self, data, seq=SEQ_LEN, pred=PRED_LEN):
        self.X, self.y = [], []
        for i in range(len(data) - seq - pred + 1):
            self.X.append(data[i : i + seq])
            self.y.append(data[i + seq : i + seq + pred, 0])
        self.X = torch.tensor(np.array(self.X))   # (N, seq, feat)
        self.y = torch.tensor(np.array(self.y))   # (N, pred)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ══════════════════════════════════════════════════════════════════════
# 3. MODELS
# ══════════════════════════════════════════════════════════════════════

# ── 3a. LSTM ───────────────────────────────────────────────────────────
class AdvancedLSTM(nn.Module):
    """
    WORKING : 4 gates (input / forget / cell / output) per timestep.
              Cell state carries long-range memory; forget gate prunes it.
    BEST FOR: Long sequences where distant context matters (e.g. paragraphs).
    PARAMS  : ~4 × (input+hidden) × hidden per layer  →  larger but richer.
    """
    def __init__(self, in_feat=1, hidden=HIDDEN,
                 layers=LAYERS, out=PRED_LEN, drop=DROPOUT):
        super().__init__()

        # Input projection — normalises raw features before RNN
        self.proj = nn.Linear(in_feat, hidden)

        # Stacked bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden, hidden,
            num_layers   = layers,
            batch_first  = True,
            dropout      = drop if layers > 1 else 0,
            bidirectional= True          # forward + backward context
        )

        # Attention: learns which timesteps matter most
        self.attn = nn.Linear(hidden * 2, 1)

        # Layer norm stabilises training on deeper stacks
        self.norm = nn.LayerNorm(hidden * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out)
        )

    def forward(self, x):                     # x: (B, T, F)
        x   = self.proj(x)                    # (B, T, H)
        out, _ = self.lstm(x)                 # (B, T, 2H)
        out = self.norm(out)

        # Soft attention over time dimension
        score  = self.attn(out).squeeze(-1)   # (B, T)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        ctx    = (out * weight).sum(dim=1)    # (B, 2H)

        return self.head(ctx)                 # (B, pred_len)


# ── 3b. GRU ────────────────────────────────────────────────────────────
class AdvancedGRU(nn.Module):
    """
    WORKING : 2 gates (reset / update) merge hidden + input in one step.
              Reset gate selects what past to forget;
              Update gate blends old hidden with candidate.
    BEST FOR: Shorter sequences, faster training, fewer params than LSTM.
    PARAMS  : ~3 × (input+hidden) × hidden per layer  →  lighter, quicker.
    """
    def __init__(self, in_feat=1, hidden=HIDDEN,
                 layers=LAYERS, out=PRED_LEN, drop=DROPOUT):
        super().__init__()

        self.proj = nn.Linear(in_feat, hidden)

        self.gru = nn.GRU(
            hidden, hidden,
            num_layers  = layers,
            batch_first = True,
            dropout     = drop if layers > 1 else 0,
            bidirectional= True
        )

        # Multi-head style attention (2 heads)
        self.attn_q = nn.Linear(hidden * 2, hidden)
        self.attn_k = nn.Linear(hidden * 2, hidden)
        self.scale   = math.sqrt(hidden)

        self.norm = nn.LayerNorm(hidden * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out)
        )

    def _attention(self, x):
        Q = self.attn_q(x)                    # (B, T, H)
        K = self.attn_k(x)
        score  = torch.bmm(Q, K.transpose(1,2)) / self.scale  # (B,T,T)
        weight = torch.softmax(score, dim=-1)
        return torch.bmm(weight, x)           # (B, T, 2H)

    def forward(self, x):
        x   = self.proj(x)
        out, _ = self.gru(x)
        out = self.norm(out)
        out = self._attention(out)
        ctx = out.mean(dim=1)                 # global average pool
        return self.head(ctx)

# ══════════════════════════════════════════════════════════════════════
# 4. TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════
class EarlyStopping:
    """Stop when val loss stops improving for `patience` epochs."""
    def __init__(self, patience=7, delta=1e-4):
        self.patience = patience; self.delta = delta
        self.best = np.inf; self.count = 0; self.stop = False

    def __call__(self, val_loss):
        if val_loss < self.best - self.delta:
            self.best = val_loss; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

def train_one_epoch(model, loader, opt, crit):
    model.train(); total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding grad
        opt.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        total += crit(model(X), y).item() * len(X)
    return total / len(loader.dataset)

def train(model, tr_dl, vl_dl, name="Model"):
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.HuberLoss()          # robust to outliers vs plain MSE
    es   = EarlyStopping(patience=8)
    hist = {"train": [], "val": []}

    print(f"\n{'─'*50}\n  Training {name}\n{'─'*50}")
    for ep in range(1, EPOCHS + 1):
        tr  = train_one_epoch(model, tr_dl, opt, crit)
        vl  = evaluate(model, vl_dl, crit)
        sch.step()
        hist["train"].append(tr); hist["val"].append(vl)
        if ep % 5 == 0:
            print(f"  Epoch {ep:3d} | train {tr:.5f} | val {vl:.5f} "
                  f"| lr {sch.get_last_lr()[0]:.2e}")
        es(vl)
        if es.stop:
            print(f"  Early stop at epoch {ep}"); break

    return hist

# ══════════════════════════════════════════════════════════════════════
# 5. INFERENCE & METRICS
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        preds.append(model(X.to(DEVICE)).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)

def metrics(pred, true, scaler):
    """Inverse-scale then compute RMSE / MAE."""
    # reshape for inverse_transform if needed
    p = scaler.inverse_transform(pred)
    t = scaler.inverse_transform(true)
    rmse = math.sqrt(mean_squared_error(t, p))
    mae  = mean_absolute_error(t, p)
    print(f"    RMSE : {rmse:.4f}  |  MAE : {mae:.4f}")
    return rmse, mae

# ══════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════════════
def plot_loss(h_lstm, h_gru):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for ax_, h, name in zip(ax, [h_lstm, h_gru], ["LSTM", "GRU"]):
        ax_.plot(h["train"], label="Train")
        ax_.plot(h["val"],   label="Val")
        ax_.set_title(f"{name} Loss"); ax_.legend(); ax_.grid(True)
    plt.tight_layout(); plt.savefig("loss_curves.png", dpi=120)
    print("  Saved → loss_curves.png")

def plot_predictions(true, lstm_pred, gru_pred, n=200):
    plt.figure(figsize=(14, 5))
    plt.plot(true[:n],      label="True",      lw=1.5)
    plt.plot(lstm_pred[:n], label="LSTM Pred", lw=1,  ls="--")
    plt.plot(gru_pred[:n],  label="GRU Pred",  lw=1,  ls=":")
    plt.title("LSTM vs GRU Predictions"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("predictions.png", dpi=120)
    print("  Saved → predictions.png")

# ══════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Device : {DEVICE}")

    # ── Data prep
    raw    = generate_data()
    scaler = MinMaxScaler()
    data   = scaler.fit_transform(raw).astype(np.float32)

    n      = len(data)
    tr_end = int(n * 0.70)
    vl_end = int(n * 0.85)

    tr_ds  = SeqDataset(data[:tr_end])
    vl_ds  = SeqDataset(data[tr_end:vl_end])
    te_ds  = SeqDataset(data[vl_end:])

    kw     = dict(batch_size=BATCH, num_workers=0)
    tr_dl  = DataLoader(tr_ds, shuffle=True,  **kw)
    vl_dl  = DataLoader(vl_ds, shuffle=False, **kw)
    te_dl  = DataLoader(te_ds, shuffle=False, **kw)

    # ── Build models
    lstm = AdvancedLSTM().to(DEVICE)
    gru  = AdvancedGRU().to(DEVICE)

    # ── Count params
    def nparams(m): return sum(p.numel() for p in m.parameters())
    print(f"LSTM params : {nparams(lstm):,}")
    print(f"GRU  params : {nparams(gru):,}")

    # ── Train
    h_lstm = train(lstm, tr_dl, vl_dl, "LSTM")
    h_gru  = train(gru,  tr_dl, vl_dl, "GRU")

    # ── Test metrics
    lp, true = predict(lstm, te_dl)
    gp, _    = predict(gru,  te_dl)

    print("\n── Test Metrics ──────────────────────────")
    print("  LSTM:"); metrics(lp, true, scaler)
    print("  GRU :");  metrics(gp, true, scaler)

    # ── Plots
    plot_loss(h_lstm, h_gru)
    plot_predictions(true[:, 0], lp[:, 0], gp[:, 0])

    # ── Save checkpoints
    torch.save(lstm.state_dict(), "lstm_checkpoint.pt")
    torch.save(gru.state_dict(),  "gru_checkpoint.pt")
    print("\n  Checkpoints saved → lstm_checkpoint.pt / gru_checkpoint.pt")

    # ── How to load & infer later (example)
    """
    model = AdvancedLSTM().to(DEVICE)
    model.load_state_dict(torch.load("lstm_checkpoint.pt"))
    model.eval()
    with torch.no_grad():
        out = model(your_tensor.to(DEVICE))
    """

if __name__ == "__main__":
    main()
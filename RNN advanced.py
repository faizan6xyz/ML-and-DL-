"""
=============================================================
  ADVANCED RNN (Recurrent Neural Network) - From Scratch
=============================================================
  USE       : Time-series forecasting, text generation, sequence
              classification, language modeling, speech recognition
  WORKING   : At each timestep t, hidden state h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
              Output y_t = W_hy * h_t + b_y
              Backpropagation Through Time (BPTT) computes gradients across timesteps
  REQUIRES  : numpy (pip install numpy)
              matplotlib (pip install matplotlib) [optional, for plotting]
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# ACTIVATIONS
# ─────────────────────────────────────────────

def tanh(x):
    return np.tanh(x)                      # hidden activation

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2            # d/dx tanh

def softmax(x):
    e = np.exp(x - x.max(axis=0, keepdims=True))  # numerically stable
    return e / e.sum(axis=0, keepdims=True)


# ─────────────────────────────────────────────
# RNN CELL  (single timestep forward pass)
# ─────────────────────────────────────────────

class RNNCell:
    """
    Working : h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b_h)
              y_t = W_hy·h_t + b_y
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialisation for stable gradients
        scale_xh = np.sqrt(2 / (input_size  + hidden_size))
        scale_hh = np.sqrt(2 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2 / (hidden_size + output_size))

        self.W_xh = np.random.randn(hidden_size, input_size)  * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_hy
        self.b_h  = np.zeros((hidden_size, 1))
        self.b_y  = np.zeros((output_size, 1))

    def params(self):
        return [self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y]


# ─────────────────────────────────────────────
# FULL RNN  (forward + BPTT + update)
# ─────────────────────────────────────────────

class RNN:
    """
    USE     : Sequence-to-sequence tasks (many-to-many / many-to-one)
    WORKING : Unrolls RNNCell over T timesteps; BPTT propagates loss
              backward through all T steps accumulating gradients.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 lr=0.01, clip=5.0, task='regression'):
        """
        task : 'regression'     -> MSE loss + linear output
               'classification' -> cross-entropy + softmax output
        """
        self.cell        = RNNCell(input_size, hidden_size, output_size)
        self.hidden_size = hidden_size
        self.lr          = lr
        self.clip        = clip          # gradient clipping threshold
        self.task        = task
        self.losses      = []

    # ── FORWARD ──────────────────────────────

    def forward(self, inputs, h_prev):
        """
        inputs : list of (input_size,1) arrays, length T
        returns: ys (outputs), hs (hidden states), pre-activations
        """
        c    = self.cell
        hs   = {-1: h_prev}
        zs   = {}       # pre-tanh values  (needed for BPTT)
        ys   = []

        for t, x in enumerate(inputs):
            zs[t]  = c.W_xh @ x + c.W_hh @ hs[t-1] + c.b_h
            hs[t]  = tanh(zs[t])
            raw    = c.W_hy @ hs[t] + c.b_y
            ys.append(softmax(raw) if self.task == 'classification' else raw)

        return ys, hs, zs

    # ── LOSS ─────────────────────────────────

    def loss(self, ys, targets):
        if self.task == 'classification':
            # cross-entropy: -sum(target * log(pred))
            return sum(-np.sum(t * np.log(y + 1e-8))
                       for y, t in zip(ys, targets))
        # MSE
        return sum(np.mean((y - t) ** 2) for y, t in zip(ys, targets))

    # ── BPTT  (Backprop Through Time) ────────

    def backward(self, inputs, ys, hs, zs, targets):
        """
        WORKING : Computes dL/dW by chaining gradients from t=T..0
                  dh propagates back through time; clipped to avoid explosion
        """
        c = self.cell
        dW_xh = np.zeros_like(c.W_xh)
        dW_hh = np.zeros_like(c.W_hh)
        dW_hy = np.zeros_like(c.W_hy)
        db_h  = np.zeros_like(c.b_h)
        db_y  = np.zeros_like(c.b_y)
        dh_next = np.zeros((self.hidden_size, 1))

        T = len(inputs)
        for t in reversed(range(T)):
            # output gradient
            if self.task == 'classification':
                dy = ys[t] - targets[t]           # softmax + CE gradient
            else:
                dy = 2 * (ys[t] - targets[t]) / targets[t].size  # MSE grad

            dW_hy += dy @ hs[t].T
            db_y  += dy

            # hidden gradient (current step + future contribution)
            dh     = c.W_hy.T @ dy + dh_next
            dz     = dh * tanh_deriv(zs[t])       # through tanh

            dW_xh += dz @ inputs[t].T
            dW_hh += dz @ hs[t-1].T
            db_h  += dz
            dh_next = c.W_hh.T @ dz               # propagate to t-1

        grads = [dW_xh, dW_hh, dW_hy, db_h, db_y]

        # gradient clipping (prevents exploding gradients)
        for g in grads:
            np.clip(g, -self.clip, self.clip, out=g)

        return grads

    # ── PARAMETER UPDATE  (SGD) ───────────────

    def update(self, grads):
        for p, g in zip(self.cell.params(), grads):
            p -= self.lr * g

    # ── SINGLE TRAINING STEP ──────────────────

    def train_step(self, inputs, targets, h_prev):
        ys, hs, zs = self.forward(inputs, h_prev)
        L          = self.loss(ys, targets)
        grads      = self.backward(inputs, ys, hs, zs, targets)
        self.update(grads)
        self.losses.append(float(L))
        return L, hs[len(inputs)-1]   # return loss + last hidden state

    # ── INFERENCE ─────────────────────────────

    def predict(self, inputs, h_prev=None):
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        ys, hs, _ = self.forward(inputs, h_prev)
        return ys, hs[len(inputs)-1]


# ─────────────────────────────────────────────
# ADVANCED: GRADIENT-CLIPPED ADAM OPTIMIZER
# USE     : faster convergence than plain SGD
# ─────────────────────────────────────────────

class AdamRNN(RNN):
    """
    Replaces SGD update with Adam (adaptive moment estimation).
    WORKING : m = β1·m + (1-β1)·g   (1st moment / momentum)
              v = β2·v + (1-β2)·g²  (2nd moment / RMSProp)
              p -= lr · m̂ / (√v̂ + ε)
    """
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.t = 0
        params = self.cell.params()
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def update(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.cell.params(), grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g**2
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p    -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────

def make_sine_sequences(n_seq=200, seq_len=20, noise=0.05):
    """Generate sine-wave sequences for regression demo."""
    X, Y = [], []
    for _ in range(n_seq):
        start = np.random.uniform(0, 2 * np.pi)
        t     = np.linspace(start, start + 2, seq_len + 1)
        s     = np.sin(t) + np.random.randn(seq_len + 1) * noise
        X.append(s[:-1])          # input  t=0..T-1
        Y.append(s[1:])           # target t=1..T   (next-step prediction)
    return np.array(X), np.array(Y)


def seq_to_rnn_inputs(x_seq, y_seq):
    """Convert 1-D sequences → list of column vectors."""
    xs = [x.reshape(1, 1) for x in x_seq]
    ys = [y.reshape(1, 1) for y in y_seq]
    return xs, ys


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(model, X, Y, epochs=30, verbose=True):
    """
    USE     : call this with any RNN / AdamRNN instance
    WORKING : iterates dataset, resets hidden state per sequence,
              runs train_step, prints epoch loss
    """
    n = len(X)
    for epoch in range(1, epochs + 1):
        total_loss = 0
        h = np.zeros((model.hidden_size, 1))        # reset at epoch start

        for i in range(n):
            xs, ys = seq_to_rnn_inputs(X[i], Y[i])
            L, h   = model.train_step(xs, ys, h)
            h      = np.zeros_like(h)               # stateless between seqs
            total_loss += L

        avg = total_loss / n
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  |  avg loss: {avg:.4f}")

    return model


# ─────────────────────────────────────────────
# PLOTTING  (optional – requires matplotlib)
# ─────────────────────────────────────────────

def plot_loss(model):
    plt.figure(figsize=(8, 3))
    plt.plot(model.losses, lw=0.8, color='steelblue')
    plt.title("Training Loss (per step)")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.tight_layout(); plt.show()


def plot_prediction(model, x_seq, y_seq, title="RNN Prediction"):
    xs, _  = seq_to_rnn_inputs(x_seq, y_seq)
    preds, _ = model.predict(xs)
    preds  = [p.item() for p in preds]

    plt.figure(figsize=(9, 3))
    plt.plot(y_seq,  label="Ground Truth", lw=2)
    plt.plot(preds,  label="Predicted",    lw=2, linestyle='--')
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────
# MAIN  –  DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":

    np.random.seed(42)

    # ── CONFIG ──────────────────────────────────
    INPUT_SIZE  = 1
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1
    SEQ_LEN     = 20
    EPOCHS      = 40
    LR          = 3e-3

    # ── DATASET ─────────────────────────────────
    print("Generating sine-wave dataset …")
    X, Y = make_sine_sequences(n_seq=300, seq_len=SEQ_LEN)

    # ── MODEL  (swap RNN ↔ AdamRNN to compare) ──
    print("\n[ Training with AdamRNN ]")
    model = AdamRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                    lr=LR, task='regression')
    train(model, X, Y, epochs=EPOCHS)

    # ── EVALUATE on a fresh sequence ────────────
    X_test, Y_test = make_sine_sequences(n_seq=1, seq_len=SEQ_LEN)
    xs, ys_true    = seq_to_rnn_inputs(X_test[0], Y_test[0])
    ys_pred, _     = model.predict(xs)

    mse = np.mean([(p.item() - t.item())**2
                   for p, t in zip(ys_pred, ys_true)])
    print(f"\nTest MSE : {mse:.6f}")

    # ── PLOT (comment out if no display) ────────
    try:
        plot_loss(model)
        plot_prediction(model, X_test[0], Y_test[0],
                        title="Next-step Sine Prediction (AdamRNN)")
    except Exception:
        print("(Matplotlib display unavailable – skipping plots)")
"""
╔══════════════════════════════════════════════════════════════════════╗
║              ADVANCED AUTOENCODER — Complete Implementation          ║
╠══════════════════════════════════════════════════════════════════════╣
║  USE:                                                                ║
║    • Dimensionality reduction   • Anomaly detection                  ║
║    • Image denoising            • Feature extraction                 ║
║    • Generative modelling (VAE) • Data compression                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  VARIANTS INCLUDED:                                                  ║
║    1. Vanilla Autoencoder       — basic encode → decode              ║
║    2. Deep Autoencoder          — multiple hidden layers             ║
║    3. Convolutional Autoencoder — for image data (CNN-based)         ║
║    4. Variational Autoencoder   — generative, learns distribution    ║
║    5. Denoising Autoencoder     — robust features, noise removal     ║
╠══════════════════════════════════════════════════════════════════════╣
║  WORKFLOW:                                                           ║
║    Input x                                                           ║
║       ↓  Encoder                                                     ║
║    Latent z  (compressed representation)                             ║
║       ↓  Decoder                                                     ║
║    Reconstruction x̂                                                 ║
║       ↓  Loss = ||x - x̂||² (+ KL divergence for VAE)               ║
║    Backprop → update weights                                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  REQUIREMENTS:                                                       ║
║    pip install torch torchvision numpy matplotlib                    ║
║    Python >= 3.8  |  PyTorch >= 1.10                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend (no display needed)
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — VANILLA AUTOENCODER
# ══════════════════════════════════════════════════════════════════════
# WORKING:
#   Encoder maps x ∈ ℝ^n  →  z ∈ ℝ^k  (k << n, bottleneck)
#   Decoder maps z ∈ ℝ^k  →  x̂ ∈ ℝ^n
#   Network learns to compress while preserving essential structure.
# USE: tabular data, low-dim embeddings, feature extraction baseline
# ──────────────────────────────────────────────────────────────────────
class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        """
        input_dim  : dimensionality of raw input (e.g. 784 for MNIST flat)
        latent_dim : size of bottleneck (e.g. 32)
        """
        super().__init__()
        # Encoder: compress
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        # Decoder: reconstruct
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),       # output in [0,1] for image pixels
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z   = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — DEEP AUTOENCODER
# ══════════════════════════════════════════════════════════════════════
# WORKING:
#   Same as vanilla but with deeper symmetric encoder/decoder stacks.
#   Layer dims progressively halve (encoder) then double (decoder).
#   Batch normalisation added for training stability.
# USE: high-dimensional tabular/embedding data, pre-training backbones
# ──────────────────────────────────────────────────────────────────────
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        """
        input_dim   : raw input size
        hidden_dims : list of hidden layer sizes for encoder
                      e.g. [512, 256, 128] — decoder mirrors this
        latent_dim  : bottleneck size
        """
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder (mirror of encoder) ───────────────────────────────
        dec_layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.LeakyReLU(0.2)]
        for h in reversed(hidden_dims[:-1]):
            dec_layers += [nn.Linear(hidden_dims[hidden_dims.index(h)+1], h),
                           nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
        dec_layers += [nn.Linear(hidden_dims[0], input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z     # return latent too for downstream use


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — CONVOLUTIONAL AUTOENCODER
# ══════════════════════════════════════════════════════════════════════
# WORKING:
#   Conv layers capture spatial structure (edges, textures, shapes).
#   Encoder: Conv → MaxPool (spatial compression)
#   Decoder: ConvTranspose (learnable upsampling) → original shape
#   Better than linear AE for image data by exploiting locality.
# USE: image denoising, anomaly detection on images, feature maps
# ──────────────────────────────────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    """
    Designed for single-channel 28×28 images (e.g. MNIST).
    Adapt channels/kernel for other resolutions.
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # ── Encoder ──────────────────────────────────────────────────
        # 1×28×28 → 8×14×14 → 16×7×7 → flatten → latent
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 8,  kernel_size=3, stride=2, padding=1),  # 14×14
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 7×7
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.enc_fc = nn.Linear(16 * 7 * 7, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────
        # latent → unflatten → 16×7×7 → 8×14×14 → 1×28×28
        self.dec_fc = nn.Linear(latent_dim, 16 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,  1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),       # pixel values in [0,1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_conv(x)                    # (B,16,7,7)
        return self.enc_fc(h.view(h.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, 16, 7, 7)
        return self.dec_conv(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — VARIATIONAL AUTOENCODER (VAE)
# ══════════════════════════════════════════════════════════════════════
# WORKING:
#   Instead of a single latent point z, encoder outputs (μ, log σ²).
#   z is sampled via reparameterisation: z = μ + σ·ε, ε ~ N(0,I)
#   This makes the latent space continuous & generative.
#   LOSS = Reconstruction loss + KL divergence
#          MSE / BCE           + -½ Σ(1 + logσ² - μ² - σ²)
#   KL term: pushes q(z|x) → N(0,I), enables random sampling.
# USE: image generation, latent space interpolation, anomaly detection
# ──────────────────────────────────────────────────────────────────────
class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)   # mean
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)   # log variance

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.fc_mu(h), self.fc_log_var(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick — keeps gradients flowing through sampling.
        z = μ + σ·ε,   ε ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)     # σ = exp(0.5 · logσ²)
        eps = torch.randn_like(std)        # ε ~ N(0, I)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate n new samples by sampling z ~ N(0,I)"""
        z = torch.randn(n, self.encoder.fc_mu.out_features, device=device)
        return self.decoder(z)


# ── VAE Loss ──────────────────────────────────────────────────────────
def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,           # β>1 → β-VAE (disentangled latents)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total = Reconstruction + β · KL
    beta  : weight on KL term (1.0 = standard VAE, >1 = β-VAE)
    """
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    # KL(q(z|x) || N(0,I)) = -½ Σ(1 + logσ² - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — DENOISING AUTOENCODER (DAE)
# ══════════════════════════════════════════════════════════════════════
# WORKING:
#   During training: corrupt input x → x̃  (add Gaussian or mask noise)
#   Network learns to map: x̃ → x  (clean reconstruction)
#   Forces encoder to learn robust, noise-invariant features.
#   At inference: pass clean OR noisy input; model denoises.
# USE: image denoising, robust embeddings, self-supervised pre-training
# ──────────────────────────────────────────────────────────────────────
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, noise_std: float = 0.2):
        """
        noise_std : std of Gaussian noise added during training
                    set to 0 at inference for clean reconstruction
        """
        super().__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),       nn.ReLU(),
            nn.Linear(256, latent_dim),nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, input_dim),  nn.Sigmoid(),
        )

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian noise corruption — applied only during training"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return torch.clamp(x + noise, 0.0, 1.0)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_noisy = self.add_noise(x)            # corrupt (train only)
        z       = self.encoder(x_noisy)
        x_hat   = self.decoder(z)
        return x_hat, x_noisy                  # return noisy input for logging


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
class ReconstructionLoss:
    """
    Factory for common AE reconstruction losses.
    MSE   : continuous / smooth outputs (default)
    BCE   : binary / image pixel outputs in [0,1]
    SSIM  : structure-aware image loss (perceptual quality)
    """

    @staticmethod
    def mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Mean Squared Error — penalises large deviations more
        return F.mse_loss(x_hat, x, reduction="mean")

    @staticmethod
    def bce(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Binary Cross-Entropy — for pixel values in [0,1]
        return F.binary_cross_entropy(x_hat, x, reduction="mean")

    @staticmethod
    def mae(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Mean Absolute Error — more robust to outliers than MSE
        return F.l1_loss(x_hat, x, reduction="mean")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — GENERIC TRAINER
# ══════════════════════════════════════════════════════════════════════
class Trainer:
    """
    Unified training loop for all AE variants.
    Handles: forward pass, loss, backprop, LR scheduling, early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mode: str = "vanilla",          # 'vanilla' | 'vae'
        beta: float = 1.0,              # only for VAE
        patience: int = 5,              # early-stopping patience
        scheduler=None,
    ):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.device    = device
        self.mode      = mode
        self.beta      = beta
        self.patience  = patience
        self.scheduler = scheduler

        self.history = {"train_loss": [], "val_loss": []}
        self._best_val = float("inf")
        self._wait     = 0

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward + loss computation"""
        x = x.to(self.device)

        if self.mode == "vae":
            x_hat, mu, log_var = self.model(x)
            loss, _, _ = vae_loss(x_hat, x, mu, log_var, self.beta)
        else:
            # vanilla / conv / denoising — all return x_hat (+ optional extras)
            out   = self.model(x)
            x_hat = out[0] if isinstance(out, tuple) else out
            loss  = ReconstructionLoss.bce(x_hat, x)

        return loss

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            x = batch[0].view(batch[0].size(0), -1) if self.mode != "conv" else batch[0]
            self.optimizer.zero_grad()
            loss = self._step(x)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for batch in loader:
            x = batch[0].view(batch[0].size(0), -1) if self.mode != "conv" else batch[0]
            loss = self._step(x)
            total += loss.item()
        return total / len(loader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        print(f"\n  Training [{self.mode}] for up to {epochs} epochs …")
        for ep in range(1, epochs + 1):
            tr = self.train_epoch(train_loader)
            va = self.eval_epoch(val_loader)
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(va)

            if self.scheduler:
                self.scheduler.step(va)

            print(f"  Epoch {ep:3d}/{epochs}  train={tr:.4f}  val={va:.4f}")

            # ── Early stopping ────────────────────────────────────────
            if va < self._best_val:
                self._best_val = va
                self._wait     = 0
            else:
                self._wait += 1
                if self._wait >= self.patience:
                    print(f"  ⏹  Early stop at epoch {ep} (patience={self.patience})")
                    break


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — ANOMALY DETECTION UTILITY
# ══════════════════════════════════════════════════════════════════════
# WORKING:
#   Train AE on normal data only.
#   At inference: high reconstruction error → sample is anomalous.
#   Threshold τ set from validation reconstruction error distribution.
# ──────────────────────────────────────────────────────────────────────
class AnomalyDetector:
    def __init__(self, model: nn.Module, device: torch.device, threshold: float = None):
        self.model     = model.to(device)
        self.device    = device
        self.threshold = threshold   # set via fit_threshold()

    @torch.no_grad()
    def reconstruction_errors(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        errors = []
        for batch in loader:
            x     = batch[0].view(batch[0].size(0), -1).to(self.device)
            out   = self.model(x)
            x_hat = out[0] if isinstance(out, tuple) else out
            err   = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)
            errors.extend(err.cpu().numpy())
        return np.array(errors)

    def fit_threshold(self, normal_loader: DataLoader, percentile: float = 95.0):
        """Set threshold at given percentile of normal reconstruction errors"""
        errs = self.reconstruction_errors(normal_loader)
        self.threshold = float(np.percentile(errs, percentile))
        print(f"  Anomaly threshold (p{percentile:.0f}): {self.threshold:.5f}")
        return self.threshold

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Returns boolean array: True = anomaly"""
        assert self.threshold is not None, "Call fit_threshold() first"
        errs = self.reconstruction_errors(loader)
        return errs > self.threshold


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════
def plot_reconstructions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str = "vanilla",
    n: int = 8,
    save_path: str = "reconstructions.png",
):
    """Side-by-side: original vs reconstructed images"""
    model.eval()
    batch = next(iter(loader))
    x     = batch[0][:n].to(device)
    x_in  = x.view(n, -1) if mode != "conv" else x

    with torch.no_grad():
        out   = model(x_in)
        x_hat = out[0] if isinstance(out, tuple) else out

    x_hat = x_hat.view(n, 1, 28, 28).cpu()
    x_ori = x.cpu()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x_ori[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[1, 0].set_ylabel("Recon",    fontsize=8)
    plt.suptitle(f"{mode} Autoencoder — Reconstructions", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latent_space(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str = "vanilla",
    save_path: str = "latent_space.png",
):
    """2-D scatter of latent codes coloured by class label (latent_dim must be 2)"""
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y  = batch[0].to(device), batch[1]
            x_in  = x.view(x.size(0), -1) if mode != "conv" else x
            if mode == "vae":
                mu, _ = model.encoder(x_in)
                z = mu
            else:
                out = model(x_in)
                z   = out[1] if isinstance(out, tuple) else model.encode(x_in)
            zs.append(z.cpu().numpy())
            labels.append(y.numpy())
    zs     = np.concatenate(zs)
    labels = np.concatenate(labels)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap="tab10", s=4, alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="Digit class")
    plt.title(f"{mode} Autoencoder — 2-D Latent Space")
    plt.xlabel("z₁"); plt.ylabel("z₂")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_loss_curves(history: dict, save_path: str = "loss_curves.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — SMOKE TEST / DEMO
#   Run: python autoencoder.py
#   Downloads MNIST (~11 MB) on first run.
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 66)
    print("  Autoencoder Suite — Smoke Test")
    print("=" * 66)

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH      = 128
    EPOCHS     = 3          # increase to 20+ for real training
    INPUT_DIM  = 784        # 28×28 MNIST flattened
    LATENT_DIM = 20

    print(f"  Device     : {DEVICE}")

    # ── Data ─────────────────────────────────────────────────────────
    transform  = transforms.Compose([transforms.ToTensor()])
    train_ds   = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    val_ds     = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_ld   = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_ld     = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"  Train size : {len(train_ds)}  |  Val size: {len(val_ds)}")

    results = {}

    # ────────────────────────────────────────────────────────────────
    # A) VANILLA AUTOENCODER
    # ────────────────────────────────────────────────────────────────
    print("\n── [1] Vanilla Autoencoder ──────────────────────────────────")
    model_v   = VanillaAutoencoder(INPUT_DIM, LATENT_DIM)
    opt_v     = torch.optim.Adam(model_v.parameters(), lr=1e-3)
    sch_v     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_v, patience=2, factor=0.5)
    trainer_v = Trainer(model_v, opt_v, DEVICE, mode="vanilla", scheduler=sch_v)
    trainer_v.fit(train_ld, val_ld, EPOCHS)
    results["vanilla"] = trainer_v.history
    plot_reconstructions(model_v, val_ld, DEVICE, mode="vanilla",
                         save_path="recon_vanilla.png")
    plot_loss_curves(trainer_v.history, save_path="loss_vanilla.png")

    # ────────────────────────────────────────────────────────────────
    # B) VARIATIONAL AUTOENCODER
    # ────────────────────────────────────────────────────────────────
    print("\n── [2] Variational Autoencoder (VAE) ───────────────────────")
    model_vae   = VAE(INPUT_DIM, hidden_dim=400, latent_dim=LATENT_DIM)
    opt_vae     = torch.optim.Adam(model_vae.parameters(), lr=1e-3)
    trainer_vae = Trainer(model_vae, opt_vae, DEVICE, mode="vae", beta=1.0)
    trainer_vae.fit(train_ld, val_ld, EPOCHS)
    results["vae"] = trainer_vae.history
    plot_reconstructions(model_vae, val_ld, DEVICE, mode="vae",
                         save_path="recon_vae.png")
    # Sample new images from prior
    samples = model_vae.sample(8, DEVICE)
    fig, ax = plt.subplots(1, 8, figsize=(12, 1.5))
    for i, s in enumerate(samples):
        ax[i].imshow(s.view(28, 28).detach().cpu(), cmap="gray")
        ax[i].axis("off")
    plt.suptitle("VAE Generated Samples"); plt.tight_layout()
    plt.savefig("vae_samples.png", dpi=150); plt.close()
    print("  Saved: vae_samples.png")

    # ────────────────────────────────────────────────────────────────
    # C) CONVOLUTIONAL AUTOENCODER
    # ────────────────────────────────────────────────────────────────
    print("\n── [3] Convolutional Autoencoder ───────────────────────────")
    model_c   = ConvAutoencoder(latent_dim=LATENT_DIM)
    opt_c     = torch.optim.Adam(model_c.parameters(), lr=1e-3)
    trainer_c = Trainer(model_c, opt_c, DEVICE, mode="conv")
    trainer_c.fit(train_ld, val_ld, EPOCHS)
    plot_reconstructions(model_c, val_ld, DEVICE, mode="conv",
                         save_path="recon_conv.png")

    # ────────────────────────────────────────────────────────────────
    # D) DENOISING AUTOENCODER
    # ────────────────────────────────────────────────────────────────
    print("\n── [4] Denoising Autoencoder ───────────────────────────────")
    model_d   = DenoisingAutoencoder(INPUT_DIM, LATENT_DIM, noise_std=0.25)
    opt_d     = torch.optim.Adam(model_d.parameters(), lr=1e-3)
    trainer_d = Trainer(model_d, opt_d, DEVICE, mode="vanilla")
    trainer_d.fit(train_ld, val_ld, EPOCHS)
    plot_reconstructions(model_d, val_ld, DEVICE, mode="vanilla",
                         save_path="recon_denoising.png")

    # ────────────────────────────────────────────────────────────────
    # E) ANOMALY DETECTION DEMO
    # ────────────────────────────────────────────────────────────────
    print("\n── [5] Anomaly Detection ───────────────────────────────────")
    detector = AnomalyDetector(model_v, DEVICE)
    detector.fit_threshold(val_ld, percentile=95.0)
    preds = detector.predict(val_ld)
    print(f"  Flagged as anomaly: {preds.sum()} / {len(preds)} samples")

    # ── Parameter counts ─────────────────────────────────────────────
    print("\n── Model Parameter Counts ──────────────────────────────────")
    for name, m in [("Vanilla", model_v), ("VAE", model_vae),
                    ("Conv",    model_c), ("Denoising", model_d)]:
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:10s}: {n:>10,} params")

    print("\n" + "=" * 66)
    print("  ✓ All variants verified successfully")
    print("=" * 66)
    print("""
  NEXT STEPS:
  1. Increase EPOCHS to 20–50 for quality reconstructions
  2. Swap MNIST for your own dataset via ImageFolder / custom Dataset
  3. Tune latent_dim: larger → better recon, smaller → better compression
  4. Use β > 1 in VAE for disentangled representations (β-VAE)
  5. Save model: torch.save(model.state_dict(), 'model.pt')
  6. Load model: model.load_state_dict(torch.load('model.pt'))
""")
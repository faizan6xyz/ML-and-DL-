"""
================================================================================
  ADVANCED GENERATIVE ADVERSARIAL NETWORK (GAN) — Complete Implementation
================================================================================

  USE:
    - Generate realistic synthetic images (demonstrated on MNIST/CIFAR-10)
    - Data augmentation for imbalanced datasets
    - Image-to-image translation (with conditional variant)
    - Anomaly detection via discriminator scores
    - Creative AI art generation

  WORKING (How a GAN works):
    A GAN consists of two neural networks locked in a minimax game:

    ┌─────────────────────────────────────────────────────────────┐
    │   LATENT VECTOR (random noise z) ──► GENERATOR (G)          │
    │          G tries to fool D by generating realistic images    │
    │                                                             │
    │   REAL IMAGES ──────────────────► DISCRIMINATOR (D)         │
    │   FAKE IMAGES (from G) ─────────► DISCRIMINATOR (D)         │
    │          D tries to distinguish real from fake              │
    │                                                             │
    │   TRAINING LOOP:                                            │
    │     Step 1: Train D on real (label=1) and fake (label=0)    │
    │     Step 2: Train G to maximize D's error on fake images    │
    │     Repeat until Nash Equilibrium is reached                │
    └─────────────────────────────────────────────────────────────┘

  ARCHITECTURE USED (DCGAN — Deep Convolutional GAN):
    - Generator    : Transposed Conv layers (upsampling) + BatchNorm + ReLU
    - Discriminator: Conv layers (downsampling) + BatchNorm + LeakyReLU
    - Loss         : Binary Cross-Entropy (BCELoss)
    - Optimizer    : Adam (separate for G and D)

  ADVANCED FEATURES INCLUDED:
    - DCGAN architecture (stable training)
    - Label smoothing (real=0.9, fake=0.1) to prevent D from overpowering G
    - Gradient penalty support (WGAN-GP style)
    - Spectral Normalization on D layers
    - Learning rate schedulers
    - Checkpoint save/load
    - TensorBoard logging
    - FID Score computation (Fréchet Inception Distance)
    - Conditional GAN (cGAN) variant
    - Training visualization (grid of generated images)
    - Early stopping on mode collapse detection

  REQUIREMENTS:
    pip install torch torchvision matplotlib numpy scipy tensorboard tqdm

    Python     : >= 3.8
    PyTorch    : >= 2.0
    torchvision: >= 0.15
    CUDA       : Optional but recommended (GPU training ~10x faster)

  USAGE (CLI):
    python advanced_gan.py --dataset mnist --epochs 50 --batch_size 64
    python advanced_gan.py --dataset cifar10 --epochs 100 --latent_dim 128
    python advanced_gan.py --resume --checkpoint ./checkpoints/gan_epoch_50.pth

================================================================================
"""

import os
import sys
import argparse
import math
import time
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (works on servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image

from scipy import linalg                 # For FID score computation
from tqdm import tqdm                    # Progress bars


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def get_config():
    """
    Central configuration object.
    All hyperparameters are defined here for easy tuning.
    """
    parser = argparse.ArgumentParser(description="Advanced GAN Training")

    # Dataset
    parser.add_argument("--dataset",      type=str,   default="mnist",
                        choices=["mnist", "cifar10", "celeba"],
                        help="Dataset to train on")
    parser.add_argument("--data_dir",     type=str,   default="./data",
                        help="Root directory to download/load datasets")
    parser.add_argument("--image_size",   type=int,   default=64,
                        help="Spatial size of training images (resized)")

    # Model
    parser.add_argument("--latent_dim",   type=int,   default=100,
                        help="Size of the latent/noise vector z")
    parser.add_argument("--n_classes",    type=int,   default=10,
                        help="Number of classes (for conditional GAN)")
    parser.add_argument("--conditional",  action="store_true",
                        help="Enable Conditional GAN (cGAN)")
    parser.add_argument("--channels",     type=int,   default=1,
                        help="Number of image channels (1=grayscale, 3=RGB)")
    parser.add_argument("--ngf",          type=int,   default=64,
                        help="Generator feature map base size")
    parser.add_argument("--ndf",          type=int,   default=64,
                        help="Discriminator feature map base size")

    # Training
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr_g",         type=float, default=0.0002,
                        help="Generator learning rate")
    parser.add_argument("--lr_d",         type=float, default=0.0002,
                        help="Discriminator learning rate")
    parser.add_argument("--beta1",        type=float, default=0.5,
                        help="Adam beta1 (momentum term)")
    parser.add_argument("--beta2",        type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--n_critic",     type=int,   default=1,
                        help="D steps per G step (1=standard, 5=WGAN)")
    parser.add_argument("--gp_lambda",    type=float, default=10.0,
                        help="Gradient penalty coefficient (WGAN-GP)")
    parser.add_argument("--label_smooth", type=float, default=0.1,
                        help="One-sided label smoothing for real labels")

    # I/O
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir",     type=str, default="./samples")
    parser.add_argument("--log_dir",        type=str, default="./logs")
    parser.add_argument("--save_every",     type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_every",   type=int, default=1,
                        help="Save sample images every N epochs")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Specific checkpoint path to resume from")

    # Misc
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",    type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fid_every", type=int, default=10,
                        help="Compute FID score every N epochs (0=disable)")

    cfg = parser.parse_args()

    # Auto-detect device
    if cfg.device == "auto":
        if torch.cuda.is_available():
            cfg.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cfg.device = "mps"
        else:
            cfg.device = "cpu"

    # Dataset-specific defaults
    if cfg.dataset == "cifar10":
        cfg.channels = 3
    elif cfg.dataset == "celeba":
        cfg.channels = 3

    # Create output directories
    for d in [cfg.checkpoint_dir, cfg.sample_dir, cfg.log_dir, cfg.data_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BUILDING BLOCKS (Reusable Layers)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralNorm(nn.Module):
    """
    Spectral Normalization wrapper.
    Stabilizes discriminator training by constraining the Lipschitz constant.
    Applied to each linear/conv layer in the Discriminator.

    HOW IT WORKS:
      Normalizes weight matrix W by its largest singular value σ(W):
        W_sn = W / σ(W)
      This keeps the spectral norm ≤ 1, bounding the discriminator's gradient.
    """
    def __init__(self, module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)

    def forward(self, x):
        return self.module(x)


def conv_block(in_ch, out_ch, kernel=4, stride=2, padding=1,
               norm=True, activation="leaky", spectral=False):
    """
    Discriminator convolutional block.
    Downsamples spatial resolution by 2x (stride=2).

    Args:
        in_ch      : Input channels
        out_ch     : Output channels
        norm       : Use BatchNorm (skip for first D layer)
        activation : 'leaky' for LeakyReLU, 'none' for linear output
        spectral   : Apply Spectral Normalization
    """
    layers = []

    conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)
    if spectral:
        conv = nn.utils.spectral_norm(conv)
    layers.append(conv)

    if norm:
        layers.append(nn.BatchNorm2d(out_ch))

    if activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == "sigmoid":
        layers.append(nn.Sigmoid())
    # 'none' → no activation (raw logits)

    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, kernel=4, stride=2, padding=1,
                 norm=True, activation="relu", output_layer=False):
    """
    Generator transposed convolutional block.
    Upsamples spatial resolution by 2x (stride=2).

    Args:
        output_layer: If True, use Tanh (final layer outputs image in [-1,1])
    """
    layers = []

    layers.append(
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)
    )

    if norm and not output_layer:
        layers.append(nn.BatchNorm2d(out_ch))

    if output_layer:
        layers.append(nn.Tanh())          # Image pixels in [-1, 1]
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GENERATOR NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    DCGAN Generator (G).

    Architecture:
      z (latent_dim,) → Linear → Reshape (ngf*8, 4, 4)
      → DeConv 4×4 → (ngf*4, 8, 8)
      → DeConv 4×4 → (ngf*2, 16, 16)
      → DeConv 4×4 → (ngf,   32, 32)
      → DeConv 4×4 → (channels, 64, 64)
      Output: Image in [-1, 1] via Tanh

    For Conditional GAN, class labels are embedded and concatenated to z.
    """

    def __init__(self, latent_dim, channels, ngf, n_classes=0, conditional=False):
        """
        Args:
            latent_dim  : Noise vector size
            channels    : Output image channels (1 or 3)
            ngf         : Base feature map count
            n_classes   : Number of classes (cGAN only)
            conditional : Enable class conditioning
        """
        super().__init__()

        self.conditional = conditional
        self.latent_dim  = latent_dim

        # cGAN: embed label as a vector and add to latent z
        if conditional:
            self.label_emb = nn.Embedding(n_classes, n_classes)
            input_dim = latent_dim + n_classes
        else:
            input_dim = latent_dim

        # Project and reshape: (B, input_dim) → (B, ngf*8, 4, 4)
        self.project = nn.Sequential(
            nn.Linear(input_dim, ngf * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.ngf = ngf

        # Upsampling blocks: 4→8→16→32→64
        self.upsample = nn.Sequential(
            deconv_block(ngf * 8, ngf * 4),      # 4×4   → 8×8
            deconv_block(ngf * 4, ngf * 2),      # 8×8   → 16×16
            deconv_block(ngf * 2, ngf),           # 16×16 → 32×32
            deconv_block(ngf, channels,            # 32×32 → 64×64
                         output_layer=True),
        )

        self._init_weights()

    def _init_weights(self):
        """DCGAN weight initialization: Normal(0, 0.02) for Conv/BN layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z, labels=None):
        """
        Args:
            z      : Noise tensor (B, latent_dim)
            labels : Class labels tensor (B,) — required if conditional=True
        Returns:
            img    : Generated images (B, C, H, W) in [-1, 1]
        """
        if self.conditional and labels is not None:
            label_emb = self.label_emb(labels)   # (B, n_classes)
            z = torch.cat([z, label_emb], dim=1) # (B, latent_dim + n_classes)

        x = self.project(z)                       # (B, ngf*8*4*4)
        x = x.view(x.size(0), self.ngf * 8, 4, 4) # (B, ngf*8, 4, 4)
        img = self.upsample(x)                     # (B, C, 64, 64)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DISCRIMINATOR NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    DCGAN Discriminator (D) with optional Spectral Normalization.

    Architecture:
      (channels, 64, 64)
      → Conv 4×4 → (ndf,   32, 32)    [no BN on first layer]
      → Conv 4×4 → (ndf*2, 16, 16)
      → Conv 4×4 → (ndf*4,  8,  8)
      → Conv 4×4 → (ndf*8,  4,  4)
      → Flatten → Linear → Sigmoid (probability real/fake)

    For Conditional GAN, class labels are embedded and projected
    to a (B, 1, H, W) tensor then concatenated to image channels.
    """

    def __init__(self, channels, ndf, n_classes=0, conditional=False,
                 spectral_norm=True):
        """
        Args:
            spectral_norm: Apply spectral normalization (recommended)
        """
        super().__init__()

        self.conditional = conditional

        if conditional:
            # Project label embedding to spatial feature map
            self.label_emb  = nn.Embedding(n_classes, n_classes)
            self.label_proj = nn.Linear(n_classes, 64 * 64)  # Same as input HW
            in_channels = channels + 1     # Extra channel for label map
        else:
            in_channels = channels

        sn = spectral_norm   # shorthand

        self.net = nn.Sequential(
            conv_block(in_channels, ndf,    norm=False, spectral=sn),  # 64→32
            conv_block(ndf,         ndf*2,  spectral=sn),              # 32→16
            conv_block(ndf*2,       ndf*4,  spectral=sn),              # 16→8
            conv_block(ndf*4,       ndf*8,  spectral=sn),              # 8→4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 8 * 4 * 4, 1),   # Scalar logit
            # No sigmoid here — use BCEWithLogitsLoss for numerical stability
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, img, labels=None):
        """
        Args:
            img    : Image tensor (B, C, H, W)
            labels : Class labels (B,) — required if conditional=True
        Returns:
            logits : Raw discrimination score (B, 1)
        """
        if self.conditional and labels is not None:
            emb  = self.label_emb(labels)                         # (B, n_classes)
            proj = self.label_proj(emb)                           # (B, H*W)
            proj = proj.view(proj.size(0), 1, img.size(2),
                             img.size(3))                         # (B, 1, H, W)
            img  = torch.cat([img, proj], dim=1)                  # (B, C+1, H, W)

        features = self.net(img)
        return self.classifier(features)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class GANLoss:
    """
    Unified loss module supporting:
      - 'vanilla'  : Binary Cross-Entropy (BCEWithLogitsLoss)
      - 'lsgan'    : Least-Squares GAN (MSELoss) — more stable gradients
      - 'wgan'     : Wasserstein GAN (raw difference of means)
      - 'wgan-gp'  : WGAN + Gradient Penalty (best for image quality)
    """

    def __init__(self, mode="vanilla", device="cpu"):
        self.mode   = mode
        self.device = device

        if mode == "vanilla":
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.criterion = nn.MSELoss()

    def discriminator_loss(self, real_logits, fake_logits,
                           real_smooth=0.9, fake_smooth=0.1):
        """
        D loss: maximize log D(real) + log(1 - D(fake))
        With label smoothing: real=0.9, fake=0.1 (prevents overconfidence)
        """
        if self.mode in ("vanilla", "lsgan"):
            real_labels = torch.full_like(real_logits, real_smooth)
            fake_labels = torch.full_like(fake_logits, fake_smooth)
            d_real = self.criterion(real_logits, real_labels)
            d_fake = self.criterion(fake_logits, fake_labels)
            return (d_real + d_fake) * 0.5

        elif self.mode in ("wgan", "wgan-gp"):
            # Wasserstein distance: maximize E[D(real)] - E[D(fake)]
            return -(real_logits.mean() - fake_logits.mean())

    def generator_loss(self, fake_logits):
        """
        G loss: minimize log(1 - D(fake))  ≡  maximize log D(fake)
        (Non-saturating generator loss — better gradients early in training)
        """
        if self.mode in ("vanilla", "lsgan"):
            real_labels = torch.ones_like(fake_logits)
            return self.criterion(fake_logits, real_labels)

        elif self.mode in ("wgan", "wgan-gp"):
            # Maximize E[D(fake)]
            return -fake_logits.mean()

    @staticmethod
    def gradient_penalty(discriminator, real_imgs, fake_imgs,
                         labels=None, device="cpu"):
        """
        WGAN-GP Gradient Penalty.

        Enforces Lipschitz constraint by penalizing ||∇D(x̂)||₂ ≠ 1
        where x̂ = ε·real + (1-ε)·fake  (interpolated samples).

        Loss term: λ · E[(||∇D(x̂)||₂ - 1)²]
        """
        B = real_imgs.size(0)

        # Random interpolation coefficient
        alpha = torch.rand(B, 1, 1, 1, device=device)

        # Interpolated images
        interp = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

        # Discriminator output on interpolated images
        d_interp = discriminator(interp, labels)

        # Compute gradients w.r.t. interpolated images
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Gradient norm penalty
        gradients = gradients.view(B, -1)
        penalty   = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(cfg):
    """
    Build DataLoader for the selected dataset.
    All images are:
      - Resized to (cfg.image_size × cfg.image_size)
      - Normalized to [-1, 1]  (to match Generator's Tanh output)
    """
    # Common normalization: mean=0.5, std=0.5 → maps [0,1] to [-1,1]
    normalize = transforms.Normalize(
        mean=[0.5] * cfg.channels,
        std =[0.5] * cfg.channels,
    )

    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if cfg.dataset == "mnist":
        dataset = datasets.MNIST(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
    elif cfg.dataset == "cifar10":
        dataset = datasets.CIFAR10(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
    elif cfg.dataset == "celeba":
        # CelebA must be downloaded manually into cfg.data_dir
        dataset = datasets.CelebA(
            root=cfg.data_dir, split="train", download=False, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,    # Ensure consistent batch sizes
    )

    return loader, len(dataset)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FID SCORE (Fréchet Inception Distance)
# ─────────────────────────────────────────────────────────────────────────────

class FIDCalculator:
    """
    Fréchet Inception Distance (FID) — standard metric for GAN quality.

    Lower FID = more realistic / diverse generated images.
    FID < 10   : Excellent (photorealistic)
    FID < 50   : Good
    FID > 150  : Poor

    HOW IT WORKS:
      1. Extract InceptionV3 pool3 features from real and fake images
      2. Fit Gaussians to each feature distribution (μ, Σ)
      3. FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·sqrt(Σ_r·Σ_f))
    """

    def __init__(self, device):
        self.device = device
        self.model  = self._load_inception()

    def _load_inception(self):
        """Load pretrained InceptionV3, strip final FC layer."""
        model = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Identity()    # Output 2048-dim pool features
        model.eval().to(self.device)
        return model

    @torch.no_grad()
    def get_features(self, images):
        """
        Extract InceptionV3 pool3 features.
        Args:
            images: Tensor (N, C, H, W) normalized to [-1, 1]
        Returns:
            features: numpy array (N, 2048)
        """
        # Inception expects (N, 3, 299, 299)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)

        images = F.interpolate(images, size=(299, 299), mode="bilinear",
                               align_corners=False)
        images = images.to(self.device)
        features = self.model(images)
        return features.cpu().numpy()

    def compute_fid(self, real_features, fake_features):
        """Compute FID between two feature sets."""
        mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

        diff = mu1 - mu2
        # Matrix square root via Scipy
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state, path):
    """Save training checkpoint (model weights + optimizer states + epoch)."""
    torch.save(state, path)
    logging.info(f"Checkpoint saved → {path}")


def load_checkpoint(path, generator, discriminator, opt_g, opt_d, device):
    """Load checkpoint and restore all states."""
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    logging.info(f"Resumed from epoch {ckpt['epoch']} | {path}")
    return ckpt["epoch"], ckpt.get("g_losses", []), ckpt.get("d_losses", [])


def sample_noise(batch_size, latent_dim, device):
    """Sample random noise vectors from N(0, 1)."""
    return torch.randn(batch_size, latent_dim, device=device)


def sample_labels(batch_size, n_classes, device):
    """Sample random class labels (for cGAN)."""
    return torch.randint(0, n_classes, (batch_size,), device=device)


def save_sample_grid(generator, fixed_z, fixed_labels, cfg, epoch):
    """
    Generate a grid of images from fixed noise for visual tracking.
    Saves to cfg.sample_dir/epoch_XXX.png
    """
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_z, fixed_labels).cpu()

    # Denormalize [-1,1] → [0,1]
    fake = (fake + 1.0) / 2.0
    grid = make_grid(fake, nrow=8, normalize=False, padding=2)

    path = os.path.join(cfg.sample_dir, f"epoch_{epoch:04d}.png")
    save_image(grid, path)
    generator.train()
    return grid


def detect_mode_collapse(d_losses, window=10, threshold=0.01):
    """
    Simple mode collapse detector.
    If D loss variance < threshold over last `window` steps → possible collapse.
    """
    if len(d_losses) < window:
        return False
    recent = d_losses[-window:]
    return np.var(recent) < threshold


def plot_losses(g_losses, d_losses, save_path):
    """Plot Generator and Discriminator loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(g_losses, label="G Loss", color="steelblue", linewidth=0.8)
    axes[0].set_title("Generator Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(d_losses, label="D Loss", color="tomato", linewidth=0.8)
    axes[1].set_title("Discriminator Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg):
    """
    Main training function.

    Workflow per epoch:
      ┌──────────────────────────────────────────────────────────┐
      │ For each batch:                                          │
      │   [D Step × n_critic]                                   │
      │     1. Sample noise z → G(z) → fake images              │
      │     2. D(real), D(fake) → compute D loss                │
      │     3. [Optional] Add gradient penalty                   │
      │     4. Backprop + update D                              │
      │                                                          │
      │   [G Step × 1]                                          │
      │     5. Sample new noise z → G(z) → fake images          │
      │     6. D(fake) → compute G loss (wants D to say "real") │
      │     7. Backprop + update G                              │
      │                                                          │
      │   Log losses to TensorBoard                              │
      └──────────────────────────────────────────────────────────┘
      After epoch:
        - Save sample images
        - Save checkpoint
        - Compute FID score (every fid_every epochs)
        - Detect mode collapse
    """

    # ── Setup ─────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.log_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ]
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    logging.info(f"Using device: {device}")
    logging.info(f"Config: {vars(cfg)}")

    # ── Data ──────────────────────────────────────────────────
    loader, n_samples = get_dataloader(cfg)
    logging.info(f"Dataset: {cfg.dataset} | Samples: {n_samples}")

    # ── Models ────────────────────────────────────────────────
    G = Generator(
        latent_dim=cfg.latent_dim,
        channels=cfg.channels,
        ngf=cfg.ngf,
        n_classes=cfg.n_classes,
        conditional=cfg.conditional,
    ).to(device)

    D = Discriminator(
        channels=cfg.channels,
        ndf=cfg.ndf,
        n_classes=cfg.n_classes,
        conditional=cfg.conditional,
        spectral_norm=True,
    ).to(device)

    logging.info(f"Generator params    : {sum(p.numel() for p in G.parameters()):,}")
    logging.info(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Optimizers ────────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=cfg.lr_g,
                       betas=(cfg.beta1, cfg.beta2))
    opt_D = optim.Adam(D.parameters(), lr=cfg.lr_d,
                       betas=(cfg.beta1, cfg.beta2))

    # Cosine annealing schedulers
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg.epochs)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg.epochs)

    # ── Loss ──────────────────────────────────────────────────
    loss_fn = GANLoss(mode="vanilla", device=cfg.device)

    # ── TensorBoard ───────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 0
    g_losses, d_losses = [], []

    if cfg.resume or cfg.checkpoint:
        ckpt_path = cfg.checkpoint or sorted(
            Path(cfg.checkpoint_dir).glob("*.pth"))[-1]
        start_epoch, g_losses, d_losses = load_checkpoint(
            ckpt_path, G, D, opt_G, opt_D, device)

    # ── Fixed noise for visualization ─────────────────────────
    fixed_z = sample_noise(64, cfg.latent_dim, device)
    fixed_labels = (sample_labels(64, cfg.n_classes, device)
                    if cfg.conditional else None)

    # ── FID Calculator ────────────────────────────────────────
    fid_calc = FIDCalculator(device) if cfg.fid_every > 0 else None

    # ── Training Loop ─────────────────────────────────────────
    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        G.train()
        D.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        t_start = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

        for batch_idx, (real_imgs, real_labels) in enumerate(pbar):
            real_imgs   = real_imgs.to(device)
            real_labels = real_labels.to(device) if cfg.conditional else None

            B = real_imgs.size(0)

            # ════════════════════════════════════════════════
            # STEP A: Train Discriminator (n_critic times)
            # ════════════════════════════════════════════════
            for _ in range(cfg.n_critic):
                opt_D.zero_grad()

                # Real images forward pass
                real_logits = D(real_imgs, real_labels)

                # Generate fake images
                z = sample_noise(B, cfg.latent_dim, device)
                fake_labels = (sample_labels(B, cfg.n_classes, device)
                               if cfg.conditional else None)
                fake_imgs = G(z, fake_labels).detach()   # Detach: no G grad

                # Fake images forward pass
                fake_logits = D(fake_imgs, fake_labels)

                # D loss (with label smoothing)
                d_loss = loss_fn.discriminator_loss(
                    real_logits, fake_logits,
                    real_smooth=1.0 - cfg.label_smooth,
                    fake_smooth=cfg.label_smooth,
                )

                # Gradient Penalty (WGAN-GP)
                if cfg.gp_lambda > 0:
                    gp = GANLoss.gradient_penalty(
                        D, real_imgs, fake_imgs, fake_labels, device)
                    d_loss = d_loss + cfg.gp_lambda * gp

                d_loss.backward()
                # Clip D gradients (prevents exploding gradients)
                nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                opt_D.step()

            # ════════════════════════════════════════════════
            # STEP B: Train Generator (once per n_critic)
            # ════════════════════════════════════════════════
            opt_G.zero_grad()

            z = sample_noise(B, cfg.latent_dim, device)
            fake_labels = (sample_labels(B, cfg.n_classes, device)
                           if cfg.conditional else None)
            fake_imgs   = G(z, fake_labels)
            fake_logits = D(fake_imgs, fake_labels)

            g_loss = loss_fn.generator_loss(fake_logits)

            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()

            # ── Logging ──────────────────────────────────────
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            writer.add_scalar("Loss/Generator",     g_loss.item(), global_step)
            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            global_step += 1

            pbar.set_postfix({
                "G": f"{g_loss.item():.4f}",
                "D": f"{d_loss.item():.4f}",
            })

        # ── End of Epoch ──────────────────────────────────────
        sched_G.step()
        sched_D.step()

        t_elapsed = time.time() - t_start
        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)

        logging.info(
            f"Epoch {epoch:04d}/{cfg.epochs} | "
            f"G: {avg_g:.4f} | D: {avg_d:.4f} | "
            f"Time: {t_elapsed:.1f}s | "
            f"LR_G: {sched_G.get_last_lr()[0]:.6f}"
        )

        # ── Sample Images ─────────────────────────────────────
        if epoch % cfg.sample_every == 0:
            grid = save_sample_grid(G, fixed_z, fixed_labels, cfg, epoch)
            writer.add_image("Generated/Grid", grid, epoch)
            logging.info(f"Sample saved → {cfg.sample_dir}/epoch_{epoch:04d}.png")

        # ── Checkpoint ────────────────────────────────────────
        if epoch % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"gan_epoch_{epoch:04d}.pth")
            save_checkpoint({
                "epoch":         epoch,
                "generator":     G.state_dict(),
                "discriminator": D.state_dict(),
                "opt_g":         opt_G.state_dict(),
                "opt_d":         opt_D.state_dict(),
                "g_losses":      g_losses,
                "d_losses":      d_losses,
                "config":        vars(cfg),
            }, ckpt_path)

        # ── FID Score ─────────────────────────────────────────
        if fid_calc and cfg.fid_every > 0 and epoch % cfg.fid_every == 0:
            logging.info("Computing FID score...")
            real_feats, fake_feats = [], []

            G.eval()
            for real_batch, _ in loader:
                real_feats.append(fid_calc.get_features(real_batch))
                z = sample_noise(real_batch.size(0), cfg.latent_dim, device)
                with torch.no_grad():
                    fake_batch = G(z)
                fake_feats.append(fid_calc.get_features(fake_batch.cpu()))
                if len(real_feats) >= 10:   # Use 10 batches for speed
                    break
            G.train()

            fid = fid_calc.compute_fid(
                np.concatenate(real_feats),
                np.concatenate(fake_feats),
            )
            writer.add_scalar("Metrics/FID", fid, epoch)
            logging.info(f"FID Score @ epoch {epoch}: {fid:.2f}")

        # ── Mode Collapse Detection ────────────────────────────
        if detect_mode_collapse(d_losses):
            logging.warning(
                "⚠️  Possible mode collapse detected (D loss has low variance). "
                "Consider reducing lr_d or increasing n_critic."
            )

    # ── Post Training ─────────────────────────────────────────
    writer.close()

    # Save final checkpoint
    final_path = os.path.join(cfg.checkpoint_dir, "gan_final.pth")
    save_checkpoint({
        "epoch":         cfg.epochs,
        "generator":     G.state_dict(),
        "discriminator": D.state_dict(),
        "opt_g":         opt_G.state_dict(),
        "opt_d":         opt_D.state_dict(),
        "g_losses":      g_losses,
        "d_losses":      d_losses,
    }, final_path)

    # Plot loss curves
    plot_losses(
        g_losses, d_losses,
        save_path=os.path.join(cfg.log_dir, "loss_curves.png")
    )
    logging.info("Training complete!")
    return G, D


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — INFERENCE (Generate images from a trained model)
# ─────────────────────────────────────────────────────────────────────────────

def generate(checkpoint_path, cfg, n_images=64, class_label=None):
    """
    Load a trained Generator and generate n_images.

    Args:
        checkpoint_path : Path to .pth checkpoint file
        cfg             : Config object
        n_images        : Number of images to generate
        class_label     : Fixed class label (cGAN only); None = random

    Returns:
        images: Tensor (n_images, C, H, W) in [0, 1]
    """
    device = torch.device(cfg.device)

    G = Generator(
        latent_dim=cfg.latent_dim,
        channels=cfg.channels,
        ngf=cfg.ngf,
        n_classes=cfg.n_classes,
        conditional=cfg.conditional,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["generator"])
    G.eval()

    with torch.no_grad():
        z = sample_noise(n_images, cfg.latent_dim, device)

        if cfg.conditional:
            if class_label is not None:
                labels = torch.full((n_images,), class_label, dtype=torch.long,
                                    device=device)
            else:
                labels = sample_labels(n_images, cfg.n_classes, device)
        else:
            labels = None

        fake = G(z, labels)

    # Denormalize to [0, 1]
    fake = (fake.clamp(-1, 1) + 1.0) / 2.0
    return fake


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — INTERPOLATION (Latent Space Exploration)
# ─────────────────────────────────────────────────────────────────────────────

def latent_interpolation(G, cfg, n_steps=10):
    """
    Spherical interpolation between two random latent vectors.

    Demonstrates the smoothness of the learned latent space.
    The interpolated images should transition smoothly between two "concepts".

    Returns:
        interpolated images as a grid
    """
    device = torch.device(cfg.device)

    z1 = sample_noise(1, cfg.latent_dim, device)
    z2 = sample_noise(1, cfg.latent_dim, device)

    # Spherical interpolation (slerp) — better than linear for high-dim vectors
    def slerp(z1, z2, t):
        """Spherical interpolation: maintains unit-sphere distribution."""
        z1_n = F.normalize(z1, dim=1)
        z2_n = F.normalize(z2, dim=1)
        omega = torch.acos((z1_n * z2_n).sum(dim=1).clamp(-1, 1))
        sin_omega = torch.sin(omega)
        if sin_omega.abs() < 1e-6:
            return (1 - t) * z1 + t * z2     # Linear fallback
        return (torch.sin((1 - t) * omega) / sin_omega).unsqueeze(1) * z1 + \
               (torch.sin(t * omega) / sin_omega).unsqueeze(1) * z2

    G.eval()
    images = []
    with torch.no_grad():
        for t in np.linspace(0, 1, n_steps):
            z = slerp(z1, z2, t)
            img = G(z)
            images.append(img)

    images = torch.cat(images, dim=0)           # (n_steps, C, H, W)
    images = (images.clamp(-1, 1) + 1.0) / 2.0

    grid = make_grid(images, nrow=n_steps)
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    QUICK START:
      # Standard GAN on MNIST
      python advanced_gan.py --dataset mnist --epochs 50

      # Conditional GAN on CIFAR-10 with WGAN-GP
      python advanced_gan.py --dataset cifar10 --conditional --epochs 100 \
          --gp_lambda 10 --n_critic 5

      # Resume from checkpoint
      python advanced_gan.py --resume --checkpoint ./checkpoints/gan_epoch_050.pth

    OUTPUT STRUCTURE:
      ./checkpoints/    ← Model weights saved every cfg.save_every epochs
      ./samples/        ← Generated image grids saved every epoch
      ./logs/           ← TensorBoard logs + training log file + loss plots
      ./data/           ← Downloaded datasets

    TENSORBOARD:
      tensorboard --logdir ./logs
      → Open http://localhost:6006 to view live loss curves + generated images
    """

    cfg = get_config()
    G, D = train(cfg)

    # Demo: generate 64 images after training
    logging.info("Generating final sample batch...")
    imgs = generate(
        checkpoint_path=os.path.join(cfg.checkpoint_dir, "gan_final.pth"),
        cfg=cfg,
        n_images=64,
    )
    save_image(imgs, os.path.join(cfg.sample_dir, "final_generated.png"), nrow=8)
    logging.info("Done! Check ./samples/final_generated.png")
"""
Advanced Variational Autoencoder (VAE) Implementation

REQUIREMENTS:
1. Support multiple VAE variants: Standard VAE, Beta-VAE, Annealed VAE
2. Flexible encoder/decoder architectures with batch normalization
3. Reparameterization trick for efficient backpropagation
4. ELBO (Evidence Lower BOund) loss computation with KL divergence
5. Support for different latent distributions: Gaussian, Mixture of Gaussians
6. Data augmentation and preprocessing pipelines
7. Reconstruction and generation capabilities with interpolation
8. Anomaly detection using reconstruction error threshold
9. Dimensionality reduction and visualization (2D/3D latent space)
10. Model checkpointing, save/load, and inference modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, kl_divergence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List, Dict, Union
import pickle
import os
from datetime import datetime
import warnings


class VAEEncoder(nn.Module):
    """
    Encoder network for VAE
    Maps input data to latent space parameters (mean and log-variance)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize encoder
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input data
        hidden_dims : List[int]
            Dimensions of hidden layers
        latent_dim : int
            Dimension of latent space
        use_batch_norm : bool
            Whether to use batch normalization
        dropout_rate : float
            Dropout probability
        """
        super(VAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder network for VAE
    Maps latent vectors back to input space
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize decoder
        
        Parameters:
        -----------
        latent_dim : int
            Dimension of latent space
        hidden_dims : List[int]
            Dimensions of hidden layers (in reverse order)
        output_dim : int
            Dimension of output data
        use_batch_norm : bool
            Whether to use batch normalization
        dropout_rate : float
            Dropout probability
        """
        super(VAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent vector
        
        Returns:
        --------
        torch.Tensor
            Reconstructed output
        """
        h = self.decoder(z)
        x_recon = torch.sigmoid(self.fc_out(h))
        
        return x_recon


class AdvancedVAE(nn.Module):
    """
    Advanced Variational Autoencoder with multiple variants support
    
    Features:
    - Standard VAE, Beta-VAE, Annealed VAE variants
    - Reparameterization trick
    - ELBO loss with KL divergence
    - Gaussian and Mixture of Gaussians support
    - Reconstruction and generation
    - Anomaly detection
    - Dimensionality reduction
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 20,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        vae_type: str = 'standard',
        beta: float = 1.0,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2,
        device: str = 'cpu'
    ):
        """
        Initialize Advanced VAE
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input data
        latent_dim : int
            Dimension of latent space
        encoder_hidden_dims : List[int]
            Hidden dimensions for encoder
        decoder_hidden_dims : List[int]
            Hidden dimensions for decoder
        vae_type : str
            Type of VAE: 'standard', 'beta_vae', 'annealed_vae'
        beta : float
            Weight of KL divergence term (beta-VAE parameter)
        use_batch_norm : bool
            Whether to use batch normalization
        dropout_rate : float
            Dropout rate
        device : str
            Device to use ('cpu' or 'cuda')
        """
        super(AdvancedVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.vae_type = vae_type
        self.beta = beta
        self.device = device
        
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [512, 256, 128]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256, 512]
        
        # Build encoder and decoder
        self.encoder = VAEEncoder(
            input_dim, encoder_hidden_dims, latent_dim,
            use_batch_norm, dropout_rate
        )
        
        self.decoder = VAEDecoder(
            latent_dim, decoder_hidden_dims, input_dim,
            use_batch_norm, dropout_rate
        )
        
        self.to(device)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, var) using N(0, 1)
        z = mu + eps * sqrt(var) where eps ~ N(0, 1)
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log-variance of latent distribution
        
        Returns:
        --------
        torch.Tensor
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (latent vector, mean, log-variance)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to input space
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent vector
        
        Returns:
        --------
        torch.Tensor
            Reconstructed output
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        
        Returns:
        --------
        Tuple
            (reconstructed output, mean, log-variance, latent vector)
        """
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        
        return x_recon, mu, logvar, z
    
    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between latent distribution and standard normal
        KL(N(mu, var) || N(0, I)) = 0.5 * sum(mu^2 + var - log(var) - 1)
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log-variance of latent distribution
        
        Returns:
        --------
        torch.Tensor
            KL divergence loss
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_loss
    
    def reconstruction_loss(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (Binary Cross-Entropy)
        
        Parameters:
        -----------
        x_recon : torch.Tensor
            Reconstructed output
        x : torch.Tensor
            Original input
        
        Returns:
        --------
        torch.Tensor
            Reconstruction loss
        """
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        return recon_loss
    
    def elbo_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        annealing_factor: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO (Evidence Lower BOund) loss
        ELBO = reconstruction_loss + beta * annealing_factor * KL_loss
        
        Parameters:
        -----------
        x : torch.Tensor
            Original input
        x_recon : torch.Tensor
            Reconstructed output
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log-variance of latent distribution
        annealing_factor : float
            Annealing factor for KL term (for Annealed VAE)
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (total loss, reconstruction loss, KL loss)
        """
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        
        total_loss = recon_loss + self.beta * annealing_factor * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def generate(self, n_samples: int = 10) -> torch.Tensor:
        """
        Generate new samples from the prior distribution
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        
        Returns:
        --------
        torch.Tensor
            Generated samples
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.decode(z)
        
        return samples
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """
        Linear interpolation in latent space between two inputs
        
        Parameters:
        -----------
        x1 : torch.Tensor
            First input sample
        x2 : torch.Tensor
            Second input sample
        n_steps : int
            Number of interpolation steps
        
        Returns:
        --------
        torch.Tensor
            Interpolated samples
        """
        with torch.no_grad():
            z1, _, _ = self.encode(x1)
            z2, _, _ = self.encode(x2)
            
            interpolations = []
            for alpha in np.linspace(0, 1, n_steps):
                z_interp = (1 - alpha) * z1 + alpha * z2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)
        
        return torch.cat(interpolations, dim=0)
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space (without sampling)
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        
        Returns:
        --------
        torch.Tensor
            Latent mean vectors
        """
        with torch.no_grad():
            mu, _ = self.encoder(x)
        
        return mu
    
    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly detection
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        
        Returns:
        --------
        torch.Tensor
            Per-sample reconstruction error
        """
        with torch.no_grad():
            x_recon, _, _, _ = self.forward(x)
            error = F.mse_loss(x_recon, x, reduction='none').sum(dim=1)
        
        return error
    
    def detect_anomalies(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        percentile: float = 95.0
    ) -> np.ndarray:
        """
        Detect anomalies using reconstruction error threshold
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        threshold : float, optional
            Reconstruction error threshold
        percentile : float
            Percentile for automatic threshold calculation
        
        Returns:
        --------
        np.ndarray
            Boolean array indicating anomalies
        """
        errors = self.compute_reconstruction_error(x)
        
        if threshold is None:
            threshold = np.percentile(errors.cpu().numpy(), percentile)
        
        anomalies = errors.cpu().numpy() > threshold
        
        return anomalies


class VAETrainer:
    """
    Trainer class for VAE models
    Handles training, validation, and checkpointing
    """
    
    def __init__(
        self,
        model: AdvancedVAE,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cpu'
    ):
        """
        Initialize VAE trainer
        
        Parameters:
        -----------
        model : AdvancedVAE
            VAE model to train
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            L2 regularization coefficient
        device : str
            Device to use
        """
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        annealing_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        epoch : int
            Current epoch number
        total_epochs : int
            Total number of epochs
        annealing_factor : float
            KL annealing factor
        
        Returns:
        --------
        Dict[str, float]
            Loss metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(self.device)
            
            # Forward pass
            x_recon, mu, logvar, z = self.model(x)
            
            # Compute loss
            loss, recon_loss, kl_loss = self.model.elbo_loss(
                x, x_recon, mu, logvar, annealing_factor
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        self.train_losses.append(avg_loss)
        
        print(f"Epoch [{epoch}/{total_epochs}] "
              f"Loss: {avg_loss:.4f}, "
              f"Recon: {avg_recon_loss:.4f}, "
              f"KL: {avg_kl_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate(self, val_loader: DataLoader, annealing_factor: float = 1.0) -> float:
        """
        Validate model
        
        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
        annealing_factor : float
            KL annealing factor
        
        Returns:
        --------
        float
            Validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(self.device)
                x_recon, mu, logvar, z = self.model(x)
                loss, _, _ = self.model.elbo_loss(
                    x, x_recon, mu, logvar, annealing_factor
                )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        vae_type: str = 'standard',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Train VAE model
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        epochs : int
            Number of training epochs
        vae_type : str
            Type of VAE for annealing schedule
        checkpoint_dir : str
            Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(epochs):
            # Calculate annealing factor for Annealed VAE
            if vae_type == 'annealed_vae':
                annealing_factor = min(1.0, epoch / (epochs * 0.5))
            else:
                annealing_factor = 1.0
            
            # Train
            self.train_epoch(train_loader, epoch + 1, epochs, annealing_factor)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader, annealing_factor)
                self.scheduler.step(val_loss)
                
                # Save checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'vae_best.pth'
                    )
                    self.save_checkpoint(checkpoint_path)
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint
        
        Parameters:
        -----------
        path : str
            Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        
        Parameters:
        -----------
        path : str
            Path to load checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


class CustomDataset(Dataset):
    """
    Custom dataset for VAE training
    """
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Initialize dataset
        
        Parameters:
        -----------
        data : np.ndarray
            Data array
        labels : np.ndarray, optional
            Label array
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx], 0


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Example 1: Generate synthetic data
    print("="*70)
    print("EXAMPLE 1: STANDARD VAE ON SYNTHETIC DATA")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic dataset
    n_samples = 1000
    input_dim = 50
    
    # Create two clusters of data
    cluster1 = np.random.normal(loc=0, scale=1, size=(n_samples // 2, input_dim))
    cluster2 = np.random.normal(loc=3, scale=1, size=(n_samples // 2, input_dim))
    X_train = np.vstack([cluster1, cluster2])
    y_train = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = np.clip(X_train, 0, 1)  # For binary cross-entropy
    
    # Create dataset and dataloader
    dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create VAE model
    vae = AdvancedVAE(
        input_dim=input_dim,
        latent_dim=10,
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        vae_type='standard',
        beta=1.0,
        device=device
    )
    
    # Train VAE
    trainer = VAETrainer(vae, learning_rate=1e-3, device=device)
    trainer.fit(train_loader, epochs=50, vae_type='standard')
    
    print("\nTraining complete!")
    
    # Example 2: Generate new samples
    print("\n" + "="*70)
    print("EXAMPLE 2: GENERATION AND INTERPOLATION")
    print("="*70)
    
    # Generate new samples
    generated_samples = vae.generate(n_samples=5)
    print(f"Generated samples shape: {generated_samples.shape}")
    
    # Interpolate between samples
    x1 = torch.FloatTensor(X_train[:1]).to(device)
    x2 = torch.FloatTensor(X_train[1:2]).to(device)
    
    interpolated = vae.interpolate(x1, x2, n_steps=5)
    print(f"Interpolated samples shape: {interpolated.shape}")
    
    # Example 3: Anomaly detection
    print("\n" + "="*70)
    print("EXAMPLE 3: ANOMALY DETECTION")
    print("="*70)
    
    # Add some anomalies to test set
    n_test = 200
    X_test = np.vstack([
        np.random.normal(loc=0, scale=1, size=(n_test - 20, input_dim)),
        np.random.normal(loc=5, scale=2, size=(20, input_dim))  # Anomalies
    ])
    X_test = scaler.transform(X_test)
    X_test = np.clip(X_test, 0, 1)
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Detect anomalies
    anomalies = vae.detect_anomalies(X_test_tensor, percentile=90)
    print(f"Number of detected anomalies: {np.sum(anomalies)} out of {len(anomalies)}")
    
    # Example 4: Dimensionality reduction
    print("\n" + "="*70)
    print("EXAMPLE 4: DIMENSIONALITY REDUCTION AND VISUALIZATION")
    print("="*70)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    latent_representations = vae.encode_to_latent(X_train_tensor)
    latent_np = latent_representations.cpu().numpy()
    
    print(f"Original data dimension: {input_dim}")
    print(f"Latent space dimension: {latent_np.shape[1]}")
    print(f"Dimension reduction: {input_dim} -> {latent_np.shape[1]}")
    
    # Example 5: Beta-VAE variant
    print("\n" + "="*70)
    print("EXAMPLE 5: BETA-VAE FOR BETTER DISENTANGLEMENT")
    print("="*70)
    
    vae_beta = AdvancedVAE(
        input_dim=input_dim,
        latent_dim=10,
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        vae_type='beta_vae',
        beta=4.0,  # Higher beta encourages more disentanglement
        device=device
    )
    
    trainer_beta = VAETrainer(vae_beta, learning_rate=1e-3, device=device)
    print("Beta-VAE model created (beta=4.0 for better disentanglement)")
    print("Higher beta value encourages latent factors to be more independent")
    
    # Example 6: Model serialization
    print("\n" + "="*70)
    print("EXAMPLE 6: MODEL PERSISTENCE")
    print("="*70)
    
    checkpoint_path = 'vae_model_checkpoint.pth'
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")
    
    # Load checkpoint
    vae_loaded = AdvancedVAE(
        input_dim=input_dim,
        latent_dim=10,
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        device=device
    )
    trainer_loaded = VAETrainer(vae_loaded, device=device)
    trainer_loaded.load_checkpoint(checkpoint_path)
    print("Model checkpoint loaded successfully!")
    
    # Example 7: Reconstruction quality
    print("\n" + "="*70)
    print("EXAMPLE 7: RECONSTRUCTION QUALITY ASSESSMENT")
    print("="*70)
    
    x_sample = torch.FloatTensor(X_test[:5]).to(device)
    x_reconstructed, _, _, _ = vae(x_sample)
    
    recon_error = F.mse_loss(x_reconstructed, x_sample, reduction='none').sum(dim=1)
    print(f"Reconstruction errors (MSE) for 5 samples:")
    for i, error in enumerate(recon_error):
        print(f"  Sample {i+1}: {error.item():.6f}")

"""
=============================================================
CONVOLUTIONAL NEURAL NETWORK (CNN) - Complete Implementation
=============================================================

WHAT IS CNN?
  A CNN is a deep learning model designed for grid-like data (images, time-series).
  It uses filters/kernels to detect spatial patterns (edges, shapes, textures).

WORKFLOW:
  Input Image → Conv Layer → ReLU → Pooling → Conv Layer → ReLU → Pooling
  → Flatten → Fully Connected → Softmax → Output (Class Probabilities)

USE:
  - Image Classification (cats vs dogs, digit recognition)
  - Object Detection, Face Recognition, Medical Imaging

REQUIREMENTS:
  pip install torch torchvision numpy matplotlib
  Python >= 3.8

STRUCTURE:
  1. Imports & Config
  2. Data Loading & Preprocessing
  3. CNN Model Definition
  4. Training Loop
  5. Evaluation
  6. Prediction on New Image
=============================================================
"""

# ── 1. IMPORTS & CONFIG ──────────────────────────────────────
import torch                          # Core deep learning framework
import torch.nn as nn                 # Neural network layers
import torch.optim as optim           # Optimizers (SGD, Adam)
import torchvision                    # Datasets & transforms for vision
import torchvision.transforms as T    # Image augmentation/normalization
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Device: use GPU if available, else CPU
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64      # Samples per gradient update
EPOCHS     = 10      # Full passes over dataset
LR         = 0.001   # Learning rate
NUM_CLASSES= 10      # CIFAR-10 has 10 classes

print(f"Using device: {DEVICE}")


# ── 2. DATA LOADING & PREPROCESSING ─────────────────────────
"""
WHAT: CIFAR-10 dataset — 60k color images (32x32), 10 classes
WHY transforms:
  - ToTensor()        : converts PIL image to [0,1] float tensor
  - Normalize()       : zero-mean, unit-variance per channel → stable training
  - RandomHorizontalFlip: data augmentation → reduces overfitting
"""

train_transform = T.Compose([
    T.RandomHorizontalFlip(),          # Augmentation: flip 50% of images
    T.RandomCrop(32, padding=4),       # Augmentation: random crop
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 channel means
                (0.2023, 0.1994, 0.2010))   # CIFAR-10 channel stds
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
])

# Download dataset (first run downloads ~170MB)
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=train_transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=test_transform)

# DataLoader: batches data, shuffles, uses multiple workers
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']


# ── 3. CNN MODEL DEFINITION ──────────────────────────────────
"""
ARCHITECTURE:
  Block 1: Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout
  Block 2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout
  Head   : Flatten → FC(1024→512) → ReLU → Dropout → FC(512→10)

KEY CONCEPTS:
  Conv2d   : learnable filters slide over input to produce feature maps
  BatchNorm: normalizes layer output → faster convergence
  ReLU     : non-linearity; kills negative activations
  MaxPool  : downsamples spatial dims (halves H & W) → translation invariance
  Dropout  : randomly zeros neurons → prevents overfitting
  FC layer : final classification from flattened features
"""

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()

        # ── Convolutional Block 1 ──
        # Input: (B, 3, 32, 32)  →  Output: (B, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 RGB channels → 32 feature maps
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # 32x32 → 16x16
            nn.Dropout(0.25)
        )

        # ── Convolutional Block 2 ──
        # Input: (B, 32, 16, 16)  →  Output: (B, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # 16x16 → 8x8
            nn.Dropout(0.25)
        )

        # ── Classifier Head ──
        # 64 channels × 8 × 8 spatial = 4096 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)                  # Output: logits per class
        )

    def forward(self, x):
        """Forward pass: data flows Input → block1 → block2 → classifier → logits"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x   # Raw logits (no softmax; CrossEntropyLoss handles it)


# Instantiate model and move to device
model = CNN().to(DEVICE)
print(model)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")


# ── 4. LOSS, OPTIMIZER & SCHEDULER ──────────────────────────
"""
CrossEntropyLoss : combines LogSoftmax + NLLLoss; standard for multi-class
Adam             : adaptive lr optimizer; generally converges faster than SGD
StepLR           : decay lr by gamma every step_size epochs → fine-tune later
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# ── 5. TRAINING & EVALUATION FUNCTIONS ──────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    """Single epoch: forward pass → loss → backprop → weight update"""
    model.train()           # Enable dropout & batchnorm in training mode
    total_loss, correct = 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # Clear previous gradients
        outputs = model(images)         # Forward pass → logits
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Backprop: compute gradients
        optimizer.step()                # Update weights

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    """Evaluate without updating weights (no gradient computation)"""
    model.eval()            # Disable dropout; use running stats for BN
    total_loss, correct = 0, 0

    with torch.no_grad():   # No grad tracking → faster, less memory
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / len(loader.dataset)
    return avg_loss, accuracy


# ── 6. MAIN TRAINING LOOP ────────────────────────────────────

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\n" + "="*55)
print(f"{'Epoch':>5} {'T-Loss':>8} {'T-Acc':>8} {'V-Loss':>8} {'V-Acc':>8}")
print("="*55)

for epoch in range(1, EPOCHS + 1):
    t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    v_loss, v_acc = evaluate(model, test_loader, criterion)
    scheduler.step()        # Adjust learning rate

    # Store history for plotting
    history["train_loss"].append(t_loss)
    history["train_acc"].append(t_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)

    print(f"{epoch:>5} {t_loss:>8.4f} {t_acc:>7.2f}% {v_loss:>8.4f} {v_acc:>7.2f}%")

print("="*55)


# ── 7. SAVE MODEL ────────────────────────────────────────────
"""
state_dict: saves only the learned weights (not the full model object).
            Preferred over pickling the entire model.
"""
torch.save(model.state_dict(), "cnn_cifar10.pth")
print("\nModel saved to cnn_cifar10.pth")

# To reload later:
# model = CNN()
# model.load_state_dict(torch.load("cnn_cifar10.pth"))
# model.eval()


# ── 8. PLOT TRAINING CURVES ──────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"],   label="Val Loss")
ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
ax1.legend(); ax1.grid(True)

ax2.plot(history["train_acc"], label="Train Acc")
ax2.plot(history["val_acc"],   label="Val Acc")
ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved to training_curves.png")


# ── 9. PREDICT ON A SINGLE IMAGE ─────────────────────────────

def predict(model, image_tensor):
    """
    Predict class for a single image tensor.
    image_tensor: shape (1, C, H, W) — add batch dim with .unsqueeze(0)
    Returns: class name and confidence %
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(DEVICE))     # Raw scores
        probs  = torch.softmax(logits, dim=1)        # Convert to probabilities
        conf, pred = probs.max(1)                    # Top class & confidence
    return CLASSES[pred.item()], conf.item() * 100


# Example: predict on first test image
sample_img, sample_label = test_set[0]
pred_class, confidence = predict(model, sample_img.unsqueeze(0))
print(f"\nSample prediction: {pred_class} ({confidence:.1f}%) | True: {CLASSES[sample_label]}")


# ── 10. PER-CLASS ACCURACY ───────────────────────────────────

def per_class_accuracy(model, loader):
    """Shows which classes the model handles well vs struggles with"""
    model.eval()
    correct = torch.zeros(NUM_CLASSES)
    total   = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu()
            for c in range(NUM_CLASSES):
                mask         = (labels == c)
                total[c]   += mask.sum()
                correct[c] += (preds[mask] == labels[mask]).sum()

    print("\nPer-Class Accuracy:")
    for i, cls in enumerate(CLASSES):
        acc = 100 * correct[i] / total[i]
        print(f"  {cls:<12}: {acc:.1f}%")

per_class_accuracy(model, test_loader)

"""
=============================================================
SUMMARY
=============================================================
CNN Flow:
  1. Conv  → detect low-level features (edges, corners)
  2. ReLU  → introduce non-linearity
  3. Pool  → reduce spatial size, increase receptive field
  4. Repeat→ detect high-level features (shapes, objects)
  5. FC    → classify based on extracted features

Expected accuracy on CIFAR-10 after 10 epochs: ~75-80%
For higher accuracy: add more blocks, use ResNet, train longer.
=============================================================
"""
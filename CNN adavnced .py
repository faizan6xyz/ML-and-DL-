"""
╔══════════════════════════════════════════════════════════════════╗
║           CONVOLUTIONAL NEURAL NETWORK (CNN) - Complete          ║
║                                                                  ║
║  USE: Image classification (e.g., CIFAR-10, custom datasets)     ║
║  WORKING:                                                        ║
║    Input → Conv+ReLU → Pool → Conv+ReLU → Pool →                ║
║    Flatten → Dense → Dropout → Softmax → Output                 ║
║  REQUIREMENTS:                                                   ║
║    pip install torch torchvision numpy matplotlib scikit-learn   ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── IMPORTS ────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time, os

# ─── CONFIG ─────────────────────────────────────────────────────────
CONFIG = {
    "batch_size"  : 64,
    "epochs"      : 20,
    "lr"          : 1e-3,
    "num_classes" : 10,          # CIFAR-10 has 10 classes
    "img_size"    : 32,          # CIFAR-10 image size
    "val_split"   : 0.1,         # 10% of train set as validation
    "seed"        : 42,
    "save_path"   : "best_cnn.pth",
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
}

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

torch.manual_seed(CONFIG["seed"])


# ─── DATA PIPELINE ──────────────────────────────────────────────────
# Transforms: normalize, augment (train only), tensorize
def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),          # augmentation
        transforms.RandomCrop(32, padding=4),       # augmentation
        transforms.ColorJitter(0.2, 0.2, 0.2),     # augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),# CIFAR-10 mean
                             (0.2470,0.2435,0.2616)),# CIFAR-10 std
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616)),
    ])
    return train_tf, test_tf

def get_dataloaders():
    train_tf, test_tf = get_transforms()

    # Downloads CIFAR-10 automatically if not present
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=train_tf)
    test_set   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=test_tf)

    # Train / Validation split
    val_len   = int(len(full_train) * CONFIG["val_split"])
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len],
                        generator=torch.Generator().manual_seed(CONFIG["seed"]))

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {train_len} | Val: {val_len} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


# ─── CNN ARCHITECTURE ───────────────────────────────────────────────
"""
BLOCK STRUCTURE:
  ConvBlock = Conv2d → BatchNorm → ReLU → (optional MaxPool)

  Input  (B, 3,  32, 32)
  Block1 (B, 32, 32, 32)  ← 2× Conv, no pool
  Block2 (B, 64, 16, 16)  ← 2× Conv + MaxPool(2)
  Block3 (B,128,  8,  8)  ← 2× Conv + MaxPool(2)
  Block4 (B,256,  4,  4)  ← 2× Conv + MaxPool(2)
  GAP    (B,256,  1,  1)  ← Global Average Pooling
  FC     (B, 512)         ← Dense + Dropout
  Out    (B, 10)          ← Softmax (implicit in CrossEntropyLoss)
"""

class ConvBlock(nn.Module):
    """Reusable Conv → BN → ReLU block"""
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),   # normalizes activations → faster training
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))   # halves spatial dims
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            ConvBlock(3,   32, pool=False),  # preserve full resolution
            ConvBlock(32,  64, pool=True),   # 32→16
            ConvBlock(64,  128, pool=True),  # 16→8
            ConvBlock(128, 256, pool=True),  # 8→4
        )

        # Global Average Pooling → removes spatial dims, robust to input size
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),             # prevents overfitting
            nn.Linear(512, num_classes), # raw logits (no softmax here)
        )

        # Weight init: Kaiming for Conv, Xavier for Linear
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)     # conv blocks
        x = self.gap(x)          # (B,256,1,1)
        x = self.classifier(x)   # (B, num_classes)
        return x                 # raw logits


# ─── TRAINING UTILITIES ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    """One full pass over training data"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping → prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluation without gradient computation"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, 100 * correct / total


# ─── TRAINING LOOP ──────────────────────────────────────────────────

def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # softens hard labels
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

    # Cosine Annealing: smoothly decays LR → avoids sharp local minima
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    history    = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_acc   = 0.0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>6}")
    print("─" * 70)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        flag = " ✓" if vl_acc > best_acc else ""
        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.2f}% | "
              f"{vl_loss:>8.4f} | {vl_acc:>6.2f}%{flag} | {lr:>8.6f} | {elapsed:>5.1f}s")

        # Save best model
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc": best_acc}, CONFIG["save_path"])

    print(f"\nBest Val Accuracy: {best_acc:.2f}%")
    return history


# ─── VISUALIZATION ──────────────────────────────────────────────────

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    plt.show()
    print("Plot saved → training_history.png")


@torch.no_grad()
def final_test(model, test_loader, device):
    """Full evaluation on held-out test set"""
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in test_loader:
        outputs = model(imgs.to(device))
        preds   = outputs.argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    print("\n── Test Set Results ──")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CLASSES)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")


# ─── INFERENCE (SINGLE IMAGE) ───────────────────────────────────────

def predict_single(model, img_tensor, device):
    """
    Predict class for one image.
    img_tensor: (3, 32, 32) normalized tensor
    Returns: (class_name, confidence%)
    """
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))    # add batch dim
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
    return CLASSES[idx.item()], conf.item() * 100


# ─── LOAD CHECKPOINT ────────────────────────────────────────────────

def load_model(path, device):
    """Resume training or run inference from saved checkpoint"""
    model = CNN(num_classes=CONFIG["num_classes"]).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%)")
    return model


# ─── MAIN ───────────────────────────────────────────────────────────

def main():
    device = CONFIG["device"]
    print(f"Device: {device.upper()}")

    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # 2. Model
    model = CNN(num_classes=CONFIG["num_classes"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(model)

    # 3. Train
    history = train(model, train_loader, val_loader, device)

    # 4. Plot training curves
    plot_history(history)

    # 5. Load best & test
    best_model = load_model(CONFIG["save_path"], device)
    final_test(best_model, test_loader, device)

    # 6. Demo inference on first test batch
    imgs, labels = next(iter(test_loader))
    name, conf = predict_single(best_model, imgs[0], device)
    print(f"\nSample Prediction → {name} ({conf:.1f}%) | True: {CLASSES[labels[0]]}")


if __name__ == "__main__":
    main()
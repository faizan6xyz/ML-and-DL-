"""
Advanced Graph Neural Network (GNN) — Full Pipeline
=====================================================
Covers:
  1. Graph construction & dataset (PyG Data objects)
  2. Multiple GNN architectures (GCN, GAT, GraphSAGE, GIN)
  3. Message passing with custom aggregation
  4. Residual connections + normalization
  5. Graph-level, node-level, link-level prediction heads
  6. Mini-batch NeighborSampler for scalable training
  7. Training loop with LR scheduler + early stopping
  8. Evaluation (accuracy, AUC, F1)
  9. ONNX export for deployment

Requirements:
    pip install torch torch_geometric ogb scikit-learn
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv,
    global_mean_pool, global_add_pool, global_max_pool,
    BatchNorm, LayerNorm,
)
from torch_geometric.utils import add_self_loops, degree, negative_sampling
from torch_geometric.loader import NeighborLoader

from sklearn.metrics import roc_auc_score, f1_score
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    # Data
    num_nodes: int = 2708          # e.g. Cora has 2708 nodes
    num_node_features: int = 1433
    num_classes: int = 7
    # Model
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.2
    heads: int = 4                 # GAT attention heads
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20             # early stopping patience
    # Sampling (for large graphs)
    num_neighbors: list = (25, 10)  # neighbors per hop
    batch_size: int = 1024
    # Task: 'node', 'graph', or 'link'
    task: str = 'node'
    # Architecture: 'gcn', 'gat', 'sage', 'gin'
    arch: str = 'gat'

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────────
# 2. GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def build_synthetic_graph() -> Data:
    """Build a random graph for demo — replace with your own data."""
    N = cfg.num_nodes
    F = cfg.num_node_features
    C = cfg.num_classes
    E = N * 5  # ~5 edges per node on average

    x = torch.randn(N, F)                           # node features
    edge_index = torch.randint(0, N, (2, E))         # random edges
    y = torch.randint(0, C, (N,))                   # node labels

    # Train / val / test masks (60 / 20 / 20)
    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:int(0.6*N)]]            = True
    val_mask  [perm[int(0.6*N):int(0.8*N)]] = True
    test_mask [perm[int(0.8*N):]]            = True

    return Data(
        x=x, edge_index=edge_index, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_nodes=N,
    )


def normalize_features(data: Data) -> Data:
    """L1-normalize each row of node features."""
    row_sum = data.x.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
    data.x = data.x / row_sum
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 3. GNN LAYER WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────

def build_conv(arch: str, in_dim: int, out_dim: int, heads: int = 1) -> nn.Module:
    """Factory for GNN convolution layers."""
    if arch == 'gcn':
        return GCNConv(in_dim, out_dim)
    elif arch == 'gat':
        # concat=False → output dim = out_dim (not out_dim*heads)
        return GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=cfg.dropout)
    elif arch == 'sage':
        return SAGEConv(in_dim, out_dim)
    elif arch == 'gin':
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        return GINConv(mlp, train_eps=True)
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose gcn/gat/sage/gin.")


# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN GNN ENCODER
# ──────────────────────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    L-layer GNN encoder with:
      - Residual connections (when dims match)
      - BatchNorm after each layer
      - Dropout for regularization
      - Optional skip connection from raw input (Jumping Knowledge)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        arch: str = 'gat',
        heads: int = 4,
        dropout: float = 0.2,
        jk: bool = True,          # Jumping Knowledge aggregation
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(build_conv(arch, hidden_dim, hidden_dim, heads=heads))
            self.norms.append(BatchNorm(hidden_dim))

        # Jumping Knowledge: aggregate all layer outputs
        if jk:
            self.jk_proj = nn.Linear(hidden_dim * (num_layers + 1), out_dim)
        else:
            self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Project raw features
        h = F.elu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        layer_outs = [h]

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual connection (same dim)
            h = h + h_new
            layer_outs.append(h)

        if self.jk:
            h = torch.cat(layer_outs, dim=-1)   # concat all hops
            h = self.jk_proj(h)
        else:
            h = self.out_proj(h)

        return h


# ──────────────────────────────────────────────────────────────────────────────
# 5. TASK-SPECIFIC HEADS
# ──────────────────────────────────────────────────────────────────────────────

class NodeClassifier(nn.Module):
    """Node-level classification."""

    def __init__(self, encoder: GNNEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.jk_proj.out_features if encoder.jk else encoder.out_proj.out_features,
                      num_classes),
        )

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.head(z)                         # [N, C] logits


class GraphClassifier(nn.Module):
    """Graph-level classification with pooling."""

    def __init__(self, encoder: GNNEncoder, num_classes: int, pool: str = 'mean'):
        super().__init__()
        self.encoder = encoder
        d = encoder.jk_proj.out_features if encoder.jk else encoder.out_proj.out_features
        self.pool_fn = {'mean': global_mean_pool, 'add': global_add_pool, 'max': global_max_pool}[pool]
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index)
        g = self.pool_fn(z, batch)                  # [B, d]
        return self.head(g)                         # [B, C] logits


class LinkPredictor(nn.Module):
    """
    Link-level prediction (dot-product decoder with MLP).
    Predicts probability of an edge between node pairs.
    """

    def __init__(self, encoder: GNNEncoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.jk_proj.out_features if encoder.jk else encoder.out_proj.out_features
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encoder(x, edge_index)

        def score(ei):
            src = z[ei[0]]
            dst = z[ei[1]]
            return self.mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)

        pos_scores = score(pos_edge_index)
        neg_scores = score(neg_edge_index)
        return pos_scores, neg_scores


# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def link_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """Binary cross-entropy for link prediction."""
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    return (pos_loss + neg_loss) / 2


# ──────────────────────────────────────────────────────────────────────────────
# 7. NODE CLASSIFICATION TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train_node_classifier(data: Data, cfg: Config):
    data = data.to(device)

    encoder = GNNEncoder(
        in_dim=data.num_node_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        arch=cfg.arch,
        heads=cfg.heads,
        dropout=cfg.dropout,
        jk=True,
    )
    model = NodeClassifier(encoder, cfg.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)
    stopper = EarlyStopping(patience=cfg.patience)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}\n")

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_pred = logits[data.val_mask].argmax(dim=-1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()

        history['train_loss'].append(loss.item())
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if stopper.step(val_acc, model):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    stopper.restore(model)

    # ── Test ──
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_pred = logits[data.test_mask].argmax(dim=-1)
        test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
        probs = F.softmax(logits[data.test_mask], dim=-1).cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()

    f1 = f1_score(y_true, test_pred.cpu().numpy(), average='macro', zero_division=0)
    print(f"\n── Final Test Results ──────────────────")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# 8. SCALABLE MINI-BATCH TRAINING (NeighborLoader)
# ──────────────────────────────────────────────────────────────────────────────

def train_with_neighbor_sampling(data: Data, cfg: Config):
    """
    For large graphs (millions of nodes): sample k neighbors per hop
    instead of full-graph message passing.
    """
    train_loader = NeighborLoader(
        data,
        num_neighbors=list(cfg.num_neighbors),
        batch_size=cfg.batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    encoder = GNNEncoder(
        in_dim=data.num_node_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.hidden_dim,
        num_layers=len(cfg.num_neighbors),
        arch=cfg.arch,
        heads=cfg.heads,
        dropout=cfg.dropout,
        jk=True,
    )
    model = NodeClassifier(encoder, cfg.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, 6):  # short demo
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Only compute loss on seed nodes (first batch_size entries)
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(
                out[:batch.batch_size],
                batch.y[:batch.batch_size],
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Mini-batch loss: {total_loss / len(train_loader):.4f}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 9. LINK PREDICTION TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_link_predictor(data: Data, cfg: Config):
    data = data.to(device)

    encoder = GNNEncoder(
        in_dim=data.num_node_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        arch='sage',               # SAGE works well for link prediction
        heads=1,
        dropout=cfg.dropout,
        jk=False,
    )
    model = LinkPredictor(encoder).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()

        # Negative sampling
        neg_edge = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
        )
        pos_scores, neg_scores = model(data.x, data.edge_index, data.edge_index, neg_edge)
        loss = link_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            # AUC on all edges
            model.eval()
            with torch.no_grad():
                p, n = model(data.x, data.edge_index, data.edge_index, neg_edge)
                scores = torch.cat([p.sigmoid(), n.sigmoid()]).cpu().numpy()
                labels = torch.cat([torch.ones(p.size(0)), torch.zeros(n.size(0))]).numpy()
                auc = roc_auc_score(labels, scores)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | AUC: {auc:.4f}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 10. ONNX EXPORT FOR DEPLOYMENT
# ──────────────────────────────────────────────────────────────────────────────

def export_to_onnx(model: nn.Module, data: Data, path: str = "gnn_model.onnx"):
    """
    Export the model for production deployment.
    Note: PyG ops need opset 17+ or custom tracing.
    This exports the head MLP only as a practical example.
    """
    model.eval()
    with torch.no_grad():
        # Get node embeddings first (graph ops can't trace directly)
        z = model.encoder(data.x.to(device), data.edge_index.to(device))
    
    head = model.head.cpu()
    z_cpu = z.cpu()
    
    torch.onnx.export(
        head,
        (z_cpu,),
        path,
        input_names=['node_embeddings'],
        output_names=['logits'],
        opset_version=17,
        dynamic_axes={'node_embeddings': {0: 'num_nodes'}},
    )
    print(f"Head exported to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 55)
    print("  Advanced GNN Pipeline")
    print("=" * 55)

    # Build and preprocess graph
    print("\n[1/4] Building synthetic graph...")
    data = build_synthetic_graph()
    data = normalize_features(data)
    print(f"  Nodes: {data.num_nodes} | Edges: {data.edge_index.size(1)} | Features: {data.num_node_features}")

    # Node classification (full-graph)
    print("\n[2/4] Node classification (full-graph, GAT)...")
    cfg.arch = 'gat'
    cfg.epochs = 50   # reduced for demo speed
    model, history = train_node_classifier(data, cfg)

    # Mini-batch training demo
    print("\n[3/4] Mini-batch NeighborLoader training (SAGE)...")
    cfg.arch = 'sage'
    _ = train_with_neighbor_sampling(data, cfg)

    # Link prediction
    print("\n[4/4] Link prediction (SAGE + negative sampling)...")
    _ = train_link_predictor(data, cfg)

    print("\nDone. Extend with your own dataset by replacing build_synthetic_graph().")
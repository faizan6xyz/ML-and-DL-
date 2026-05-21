# ==============================================================================
#   NEURAL NETWORK — COMPLETE FUNCTION & MODEL REFERENCE
#   Covers: PyTorch + TensorFlow/Keras
#   Every parameter annotated with one-line comment explaining its purpose
# ==============================================================================
# pip install torch torchvision tensorflow keras


# ==============================================================================
# SECTION 1 — PYTORCH: CORE LAYERS (torch.nn)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# LINEAR (Fully Connected / Dense) LAYER
# ------------------------------------------------------------------------------
layer = nn.Linear(
    in_features=128,          # Number of input features per sample
    out_features=64,          # Number of output features (neurons in this layer)
    bias=True,                # Add a learnable bias term to each output neuron
    device=None,              # Device to place weights on ('cpu', 'cuda', 'mps')
    dtype=None,               # Data type of weight tensors (e.g. torch.float32)
)

# ------------------------------------------------------------------------------
# CONVOLUTIONAL LAYERS
# ------------------------------------------------------------------------------

# 1D Convolution — sequences, audio, time-series
layer = nn.Conv1d(
    in_channels=1,            # Number of input channels (e.g. 1 for mono audio)
    out_channels=32,          # Number of output channels (learned filters)
    kernel_size=3,            # Width of each convolutional filter
    stride=1,                 # Step size the filter moves each time
    padding=0,                # Zero-padding added to both sides of input
    dilation=1,               # Spacing between kernel elements (dilated/atrous conv)
    groups=1,                 # Number of input/output groups (1=standard, in_channels=depthwise)
    bias=True,                # Add learnable bias to each output channel
    padding_mode='zeros',     # Padding type: 'zeros', 'reflect', 'replicate', 'circular'
    device=None,              # Target device for parameters
    dtype=None,               # Data type of parameters
)

# 2D Convolution — images
layer = nn.Conv2d(
    in_channels=3,            # Input channels (3 for RGB, 1 for grayscale)
    out_channels=64,          # Number of output feature maps (filters)
    kernel_size=3,            # Filter size: int or (height, width) tuple
    stride=1,                 # Pixels skipped per filter step: int or (h, w)
    padding=1,                # Zero-padding: int, (h,w), or 'same'/'valid'
    dilation=1,               # Dilation factor for atrous/dilated convolution
    groups=1,                 # Grouped convolution (depthwise when == in_channels)
    bias=True,                # Learnable bias added to each output channel
    padding_mode='zeros',     # How to pad: 'zeros', 'reflect', 'replicate', 'circular'
    device=None,
    dtype=None,
)

# 3D Convolution — video, volumetric medical data
layer = nn.Conv3d(
    in_channels=1,            # Input channels (depth dimension added vs Conv2d)
    out_channels=16,          # Number of output feature volumes
    kernel_size=3,            # Filter size: int or (depth, height, width)
    stride=1,                 # Step size: int or (d, h, w)
    padding=0,                # Padding: int or (d, h, w)
    dilation=1,               # Dilation for dilated 3D convolution
    groups=1,                 # Grouped convolution groups
    bias=True,                # Learnable bias per output channel
    padding_mode='zeros',
    device=None,
    dtype=None,
)

# Transposed Convolution (Deconvolution) — upsampling in decoders/GANs
layer = nn.ConvTranspose2d(
    in_channels=64,           # Number of input channels
    out_channels=32,          # Number of output channels after upsampling
    kernel_size=4,            # Filter kernel size
    stride=2,                 # Upsampling factor (stride > 1 increases spatial size)
    padding=1,                # Padding subtracted from output size
    output_padding=0,         # Extra rows/cols added to one side of output
    groups=1,                 # Grouped transposed convolution
    bias=True,                # Learnable bias term
    dilation=1,               # Dilation of input (rarely changed)
    padding_mode='zeros',
    device=None,
    dtype=None,
)

# ------------------------------------------------------------------------------
# POOLING LAYERS
# ------------------------------------------------------------------------------

# Max Pooling 2D — keeps strongest activation in each region
layer = nn.MaxPool2d(
    kernel_size=2,            # Size of the pooling window: int or (h, w)
    stride=None,              # Step size of window; defaults to kernel_size if None
    padding=0,                # Zero-padding added before pooling
    dilation=1,               # Spacing between pooling window elements
    return_indices=False,     # If True, returns indices of max values (for MaxUnpool2d)
    ceil_mode=False,          # Use ceiling instead of floor for output size calculation
)

# Average Pooling 2D — averages all activations in each region
layer = nn.AvgPool2d(
    kernel_size=2,            # Size of the averaging window
    stride=None,              # Step size; defaults to kernel_size
    padding=0,                # Zero-padding before averaging
    ceil_mode=False,          # Use ceiling for output size
    count_include_pad=True,   # Include padding zeros in average calculation
    divisor_override=None,    # Custom divisor instead of window size
)

# Adaptive Average Pooling — outputs fixed size regardless of input size
layer = nn.AdaptiveAvgPool2d(
    output_size=(1, 1),       # Target output spatial size: int or (H, W); (1,1) → GlobalAvgPool
)

# Adaptive Max Pooling
layer = nn.AdaptiveMaxPool2d(
    output_size=(7, 7),       # Target output spatial size regardless of input size
    return_indices=False,     # Return argmax indices for each output position
)

# Global Average Pooling (via AdaptiveAvgPool2d with output (1,1))
global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Collapses H×W to 1×1 per channel

# ------------------------------------------------------------------------------
# NORMALIZATION LAYERS
# ------------------------------------------------------------------------------

# Batch Normalization 1D — for fully-connected / 1D conv layers
layer = nn.BatchNorm1d(
    num_features=128,         # Number of features/channels to normalize
    eps=1e-5,                 # Small constant for numerical stability in division
    momentum=0.1,             # Running mean/var update rate (1=no moving average)
    affine=True,              # Learn scale (gamma) and shift (beta) parameters
    track_running_stats=True, # Track running mean/var for inference mode
    device=None,
    dtype=None,
)

# Batch Normalization 2D — for 2D convolutional feature maps
layer = nn.BatchNorm2d(
    num_features=64,          # Number of channels in the input feature map
    eps=1e-5,                 # Numerical stability constant added to variance
    momentum=0.1,             # Running statistics update momentum
    affine=True,              # Enable learnable affine parameters gamma and beta
    track_running_stats=True, # Maintain running mean/var used during eval
    device=None,
    dtype=None,
)

# Layer Normalization — normalizes across feature dimension (used in Transformers)
layer = nn.LayerNorm(
    normalized_shape=512,     # Shape of features to normalize: int or list [H, W, C]
    eps=1e-5,                 # Epsilon for numerical stability
    elementwise_affine=True,  # Learn per-element gamma and beta parameters
    bias=True,                # Include learnable bias (beta) term
    device=None,
    dtype=None,
)

# Group Normalization — splits channels into groups, normalizes each group
layer = nn.GroupNorm(
    num_groups=8,             # Number of groups to divide channels into
    num_channels=64,          # Total number of channels (must be divisible by num_groups)
    eps=1e-5,                 # Epsilon for numerical stability
    affine=True,              # Learn per-channel gamma and beta
    device=None,
    dtype=None,
)

# Instance Normalization — normalizes each sample independently (used in style transfer)
layer = nn.InstanceNorm2d(
    num_features=64,          # Number of channels in input
    eps=1e-5,                 # Numerical stability constant
    momentum=0.1,             # Running stats momentum (only if track_running_stats=True)
    affine=False,             # Learnable gamma/beta (usually False for style transfer)
    track_running_stats=False,# Track global running stats across batches
    device=None,
    dtype=None,
)

# ------------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# ------------------------------------------------------------------------------

# ReLU — most common; fast; avoids vanishing gradient but can "die"
relu = nn.ReLU(
    inplace=False,            # Modify input tensor in-place (saves memory, can cause issues)
)

# Leaky ReLU — allows small negative slope to prevent dying neurons
leaky = nn.LeakyReLU(
    negative_slope=0.01,      # Slope for negative inputs (typical: 0.01–0.2)
    inplace=False,            # Compute in-place to save memory
)

# PReLU — learned negative slope parameter
prelu = nn.PReLU(
    num_parameters=1,         # Number of 'a' parameters: 1=shared, or n_channels for per-channel
    init=0.25,                # Initial value of the learnable slope parameter
    device=None,
    dtype=None,
)

# ELU — smooth negative saturation; avoids dying neurons
elu = nn.ELU(
    alpha=1.0,                # Scale for negative saturation region
    inplace=False,            # In-place computation flag
)

# GELU — smooth ReLU approximation; default in Transformers (BERT, GPT)
gelu = nn.GELU(
    approximate='none',       # 'none' for exact, 'tanh' for faster approximation
)

# SiLU / Swish — self-gated activation: x * sigmoid(x); used in EfficientNet
silu = nn.SiLU(
    inplace=False,            # Compute in-place to save memory
)

# Sigmoid — squashes output to [0, 1]; used in binary classification output
sigmoid = nn.Sigmoid()        # No parameters; maps any real number to (0, 1)

# Tanh — squashes output to [-1, 1]; zero-centered
tanh = nn.Tanh()              # No parameters; maps any real number to (-1, 1)

# Softmax — converts logits to probability distribution summing to 1
softmax = nn.Softmax(
    dim=1,                    # Dimension along which softmax is computed (usually class dim)
)

# LogSoftmax — log of softmax; more numerically stable for NLLLoss
log_softmax = nn.LogSoftmax(
    dim=1,                    # Dimension along which log-softmax is computed
)

# Hardswish — efficient Swish approximation for mobile networks (MobileNetV3)
hardswish = nn.Hardswish(
    inplace=False,            # In-place operation flag
)

# Mish — self-regularized smooth activation: x * tanh(softplus(x))
mish = nn.Mish(
    inplace=False,            # In-place operation flag
)

# CELU — continuously differentiable ELU variant
celu = nn.CELU(
    alpha=1.0,                # Controls the negative saturation point
    inplace=False,
)

# Softplus — smooth approximation of ReLU; always positive output
softplus = nn.Softplus(
    beta=1,                   # Sharpness of the softplus curve
    threshold=20,             # Above this, reverts to linear for numerical stability
)

# ------------------------------------------------------------------------------
# DROPOUT LAYERS
# ------------------------------------------------------------------------------

# Standard Dropout — randomly zeroes elements during training
dropout = nn.Dropout(
    p=0.5,                    # Probability of zeroing each element (0=none, 1=all)
    inplace=False,            # Compute in-place to save memory
)

# Dropout2D — zeros entire feature maps (channels) rather than individual elements
dropout2d = nn.Dropout2d(
    p=0.5,                    # Probability of zeroing an entire channel
    inplace=False,            # In-place flag
)

# Dropout3D — zeros entire 3D feature maps during training
dropout3d = nn.Dropout3d(
    p=0.5,                    # Probability of dropping entire 3D channels
    inplace=False,
)

# AlphaDropout — dropout that preserves mean and variance; used with SELU
alpha_dropout = nn.AlphaDropout(
    p=0.5,                    # Dropout probability; maintains self-normalizing property
    inplace=False,
)

# ------------------------------------------------------------------------------
# RECURRENT LAYERS
# ------------------------------------------------------------------------------

# RNN — simple recurrent layer (suffers from vanishing gradients)
rnn = nn.RNN(
    input_size=128,           # Number of input features at each time step
    hidden_size=256,          # Number of features in hidden state (memory)
    num_layers=1,             # Number of stacked RNN layers
    nonlinearity='tanh',      # Activation: 'tanh' or 'relu'
    bias=True,                # Use bias weights
    batch_first=False,        # If True, input shape is (batch, seq, feature)
    dropout=0.0,              # Dropout between RNN layers (only if num_layers > 1)
    bidirectional=False,      # Process sequence in both forward and backward directions
    device=None,
    dtype=None,
)

# LSTM — gates control information flow; solves vanishing gradient
lstm = nn.LSTM(
    input_size=128,           # Input feature size at each time step
    hidden_size=256,          # Hidden state and cell state size
    num_layers=2,             # Number of stacked LSTM layers
    bias=True,                # Include bias terms in all gates
    batch_first=True,         # Input shape (batch, seq, features) when True
    dropout=0.3,              # Dropout rate between LSTM layers (not applied to last layer)
    bidirectional=False,      # Enable bidirectional LSTM (doubles output features)
    proj_size=0,              # If > 0, use LSTMP with projection layer of this size
    device=None,
    dtype=None,
)

# GRU — simpler gated RNN; often matches LSTM with fewer parameters
gru = nn.GRU(
    input_size=128,           # Input feature size per time step
    hidden_size=256,          # Size of the hidden state
    num_layers=1,             # Number of stacked GRU layers
    bias=True,                # Use bias in gates
    batch_first=True,         # (batch, seq, feature) input when True
    dropout=0.0,              # Dropout probability between layers
    bidirectional=False,      # Process input in both directions
    device=None,
    dtype=None,
)

# ------------------------------------------------------------------------------
# EMBEDDING LAYER
# ------------------------------------------------------------------------------
embedding = nn.Embedding(
    num_embeddings=10000,     # Vocabulary size (number of unique tokens)
    embedding_dim=256,        # Dimension of each embedding vector
    padding_idx=0,            # Index whose embedding is kept as zeros (padding token)
    max_norm=None,            # If set, normalizes embeddings with norm > max_norm
    norm_type=2.0,            # Norm type for max_norm constraint
    scale_grad_by_freq=False, # Scale gradients by inverse token frequency
    sparse=False,             # Use sparse gradient updates (faster for large vocab)
    _weight=None,             # Optional pre-loaded weight tensor
    _freeze=False,            # Freeze embedding weights (no gradient updates)
    device=None,
    dtype=None,
)

# EmbeddingBag — computes mean/sum/max of a bag of embeddings efficiently
emb_bag = nn.EmbeddingBag(
    num_embeddings=10000,     # Vocabulary size
    embedding_dim=128,        # Embedding vector dimension
    max_norm=None,            # Max norm constraint on embedding vectors
    norm_type=2.0,            # Norm type for max_norm
    scale_grad_by_freq=False, # Scale gradients by token frequency
    mode='mean',              # Aggregation: 'mean', 'sum', 'max'
    sparse=False,             # Use sparse gradients
    include_last_offset=False,# Include the last offset in offset2bag
    padding_idx=None,         # Padding index excluded from aggregation
    device=None,
    dtype=None,
)

# ------------------------------------------------------------------------------
# ATTENTION LAYERS
# ------------------------------------------------------------------------------

# Multi-Head Attention — core of Transformer architecture
attn = nn.MultiheadAttention(
    embed_dim=512,            # Total dimension of model (Q, K, V projections)
    num_heads=8,              # Number of parallel attention heads (embed_dim % num_heads == 0)
    dropout=0.1,              # Dropout on attention weights
    bias=True,                # Add bias to input/output projections
    add_bias_kv=False,        # Add bias to key and value sequences
    add_zero_attn=False,      # Add a batch of zeros to K and V sequences
    kdim=None,                # Key feature dimension (defaults to embed_dim)
    vdim=None,                # Value feature dimension (defaults to embed_dim)
    batch_first=False,        # (batch, seq, feature) input order when True
    device=None,
    dtype=None,
)

# ------------------------------------------------------------------------------
# TRANSFORMER LAYERS
# ------------------------------------------------------------------------------

# Single Transformer Encoder Layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,              # Model dimensionality (input and output size)
    nhead=8,                  # Number of attention heads
    dim_feedforward=2048,     # Hidden size of the feedforward network (typically 4×d_model)
    dropout=0.1,              # Dropout applied after attention and FFN
    activation='relu',        # FFN activation: 'relu', 'gelu', or callable
    layer_norm_eps=1e-5,      # Epsilon for layer normalization
    batch_first=False,        # (batch, seq, feature) when True
    norm_first=False,         # Pre-norm (True) vs post-norm (False) architecture
    bias=True,                # Use bias in all linear projections
    device=None,
    dtype=None,
)

# Transformer Encoder — stacks multiple encoder layers
encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,  # Base encoder layer instance to clone
    num_layers=6,             # Number of stacked encoder layers
    norm=None,                # Optional final normalization layer applied to output
    enable_nested_tensor=True,# Use nested tensors for padding mask efficiency
    mask_check=True,          # Validate mask shapes at forward pass
)

# Single Transformer Decoder Layer
decoder_layer = nn.TransformerDecoderLayer(
    d_model=512,              # Model dimensionality
    nhead=8,                  # Number of attention heads
    dim_feedforward=2048,     # FFN hidden size
    dropout=0.1,              # Dropout rate
    activation='relu',        # FFN activation function
    layer_norm_eps=1e-5,      # Layer norm epsilon
    batch_first=False,        # Input dimension order
    norm_first=False,         # Pre-layer-norm when True
    bias=True,                # Include bias in projections
    device=None,
    dtype=None,
)

# Transformer Decoder
decoder = nn.TransformerDecoder(
    decoder_layer=decoder_layer,  # Base decoder layer to clone
    num_layers=6,             # Number of stacked decoder layers
    norm=None,                # Optional output normalization layer
)

# Full Transformer (Encoder + Decoder)
transformer = nn.Transformer(
    d_model=512,              # Embedding / model dimension
    nhead=8,                  # Number of attention heads
    num_encoder_layers=6,     # Number of encoder layers
    num_decoder_layers=6,     # Number of decoder layers
    dim_feedforward=2048,     # FFN hidden size
    dropout=0.1,              # Dropout probability
    activation='relu',        # Activation for FFN: 'relu' or 'gelu'
    custom_encoder=None,      # Override with custom encoder module
    custom_decoder=None,      # Override with custom decoder module
    layer_norm_eps=1e-5,      # Layer normalization epsilon
    batch_first=False,        # Input order: (seq, batch, feat) or (batch, seq, feat)
    norm_first=False,         # Use pre-layer-norm (True) or post-layer-norm (False)
    bias=True,                # Include bias in projections
    device=None,
    dtype=None,
)

# ------------------------------------------------------------------------------
# LOSS FUNCTIONS
# ------------------------------------------------------------------------------

# MSE Loss — mean squared error for regression
loss_fn = nn.MSELoss(
    size_average=None,        # Deprecated; use 'reduction'
    reduce=None,              # Deprecated; use 'reduction'
    reduction='mean',         # 'none' per-sample, 'mean' averaged, 'sum' summed
)

# MAE / L1 Loss — mean absolute error; robust to outliers
loss_fn = nn.L1Loss(
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # 'none', 'mean', or 'sum'
)

# Huber Loss — MSE for small errors, MAE for large; robust regression
loss_fn = nn.HuberLoss(
    reduction='mean',         # 'none', 'mean', 'sum'
    delta=1.0,                # Threshold between MSE and MAE regions
)

# Smooth L1 Loss — similar to Huber; used in object detection (Faster R-CNN)
loss_fn = nn.SmoothL1Loss(
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # Output reduction method
    beta=1.0,                 # Threshold below which it is squared (like Huber's delta)
)

# Binary Cross-Entropy — binary classification with sigmoid output
loss_fn = nn.BCELoss(
    weight=None,              # Manual per-sample weight tensor for imbalanced data
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # 'none', 'mean', 'sum'
)

# BCE with Logits — combines sigmoid + BCE in numerically stable way
loss_fn = nn.BCEWithLogitsLoss(
    weight=None,              # Sample weights for imbalanced classification
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # Output aggregation method
    pos_weight=None,          # Weight for positive class (> 1 increases recall)
)

# Cross-Entropy Loss — multi-class classification
loss_fn = nn.CrossEntropyLoss(
    weight=None,              # Per-class weight tensor for class imbalance
    size_average=None,        # Deprecated
    ignore_index=-100,        # Class index to ignore in loss computation (e.g. padding)
    reduce=None,              # Deprecated
    reduction='mean',         # 'none', 'mean', 'sum'
    label_smoothing=0.0,      # Label smoothing epsilon (0=off, 0.1=common)
)

# NLL Loss — negative log-likelihood; use with LogSoftmax output
loss_fn = nn.NLLLoss(
    weight=None,              # Per-class weights
    size_average=None,        # Deprecated
    ignore_index=-100,        # Target index to ignore
    reduce=None,              # Deprecated
    reduction='mean',         # Reduction type
)

# KL Divergence — measures distribution difference; used in VAEs
loss_fn = nn.KLDivLoss(
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # 'none', 'mean', 'sum', 'batchmean' (mathematically correct)
    log_target=False,         # If True, target is in log-space
)

# CTC Loss — connectionist temporal classification for sequence-to-sequence (ASR)
loss_fn = nn.CTCLoss(
    blank=0,                  # Blank label index used in CTC algorithm
    reduction='mean',         # 'none', 'mean', 'sum'
    zero_infinity=False,      # Set infinite losses and their gradients to zero
)

# Triplet Margin Loss — metric learning: pulls anchor-positive, pushes anchor-negative
loss_fn = nn.TripletMarginLoss(
    margin=1.0,               # Minimum margin between positive and negative distances
    p=2,                      # Norm degree for distance calculation
    eps=1e-6,                 # Small value for numerical stability
    swap=False,               # Use distance swap trick for semi-hard mining
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # Output reduction
)

# Contrastive Loss — pushes similar pairs together, dissimilar apart
loss_fn = nn.CosineEmbeddingLoss(
    margin=0.0,               # Margin for negative pairs (0 = no margin)
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # 'none', 'mean', 'sum'
)

# Hinge Embedding Loss — for SVM-style learning with +1/-1 labels
loss_fn = nn.HingeEmbeddingLoss(
    margin=1.0,               # Margin threshold for negative samples
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # Reduction method
)

# ------------------------------------------------------------------------------
# OPTIMIZERS (torch.optim)
# ------------------------------------------------------------------------------

import torch.optim as optim

model_params = nn.Linear(10, 5).parameters()  # Placeholder

# SGD — classic stochastic gradient descent with optional momentum
optimizer = optim.SGD(
    params=model_params,      # Iterable of parameters or param groups to optimize
    lr=0.01,                  # Learning rate (step size for each update)
    momentum=0.9,             # Momentum factor (0=disabled); accelerates convergence
    dampening=0,              # Dampening for momentum (reduces momentum effect)
    weight_decay=1e-4,        # L2 regularization coefficient (weight decay)
    nesterov=False,           # Use Nesterov momentum (look-ahead gradient step)
    maximize=False,           # Maximize objective instead of minimize
    foreach=None,             # Use foreach implementation (faster on CUDA)
    differentiable=False,     # Enable differentiating through optimizer step
)

# Adam — adaptive learning rate; default for most deep learning
optimizer = optim.Adam(
    params=model_params,      # Model parameters to optimize
    lr=1e-3,                  # Learning rate (step size)
    betas=(0.9, 0.999),       # Decay rates for 1st (momentum) and 2nd (variance) moment estimates
    eps=1e-8,                 # Small constant for numerical stability in denominator
    weight_decay=0,           # L2 regularization (adds weight_decay * param to gradient)
    amsgrad=False,            # Use AMSGrad variant (ensures non-increasing step sizes)
    maximize=False,           # Maximize instead of minimize the objective
    foreach=None,             # Use vectorized foreach implementation
    capturable=False,         # Allow use in CUDA graph capture
    differentiable=False,     # Support differentiating through optimizer
    fused=None,               # Use fused CUDA kernel (fastest on GPU)
)

# AdamW — Adam with decoupled weight decay (recommended over Adam for transformers)
optimizer = optim.AdamW(
    params=model_params,      # Parameters to optimize
    lr=1e-3,                  # Learning rate
    betas=(0.9, 0.999),       # Decay rates for first and second moments
    eps=1e-8,                 # Numerical stability epsilon
    weight_decay=1e-2,        # Decoupled weight decay (not added to gradient)
    amsgrad=False,            # Use AMSGrad variant
    maximize=False,           # Maximize instead of minimize
    foreach=None,             # Vectorized implementation
    capturable=False,         # CUDA graph capture support
    differentiable=False,     # Differentiable through step
    fused=None,               # Fused GPU kernel
)

# RMSprop — adaptive optimizer; good for RNNs
optimizer = optim.RMSprop(
    params=model_params,      # Parameters to optimize
    lr=1e-2,                  # Learning rate
    alpha=0.99,               # Smoothing constant (decay for moving average of squared grad)
    eps=1e-8,                 # Numerical stability constant
    weight_decay=0,           # L2 regularization coefficient
    momentum=0,               # Momentum factor
    centered=False,           # Normalize gradient by estimated variance if True
    foreach=None,
    maximize=False,
    differentiable=False,
    capturable=False,
)

# Adagrad — per-parameter adaptive rates; good for sparse data / NLP
optimizer = optim.Adagrad(
    params=model_params,      # Parameters to optimize
    lr=1e-2,                  # Initial learning rate
    lr_decay=0,               # Learning rate decay over updates
    weight_decay=0,           # L2 regularization
    initial_accumulator_value=0, # Starting value for gradient accumulator
    eps=1e-10,                # Numerical stability constant
    foreach=None,
    maximize=False,
    differentiable=False,
)

# Adadelta — adaptive learning rate without manual lr; good for long training
optimizer = optim.Adadelta(
    params=model_params,      # Parameters to optimize
    lr=1.0,                   # Coefficient applied to computed update
    rho=0.9,                  # Decay factor for running average of squared gradients
    eps=1e-6,                 # Numerical stability constant
    weight_decay=0,           # L2 regularization
    foreach=None,
    capturable=False,
    maximize=False,
    differentiable=False,
)

# LBFGS — quasi-Newton optimizer; for small models and full-batch optimization
optimizer = optim.LBFGS(
    params=model_params,      # Parameters to optimize
    lr=1,                     # Learning rate
    max_iter=20,              # Max iterations per optimization step call
    max_eval=None,            # Max function evaluations per step (default: max_iter*1.25)
    tolerance_grad=1e-7,      # Stop when gradient norm below this
    tolerance_change=1e-9,    # Stop when loss change below this
    history_size=100,         # Number of past gradients stored for Hessian approximation
    line_search_fn=None,      # Line search: None or 'strong_wolfe'
)

# ------------------------------------------------------------------------------
# LEARNING RATE SCHEDULERS
# ------------------------------------------------------------------------------

dummy_opt = optim.SGD(nn.Linear(1,1).parameters(), lr=0.1)

# StepLR — multiply lr by gamma every step_size epochs
scheduler = optim.lr_scheduler.StepLR(
    optimizer=dummy_opt,      # Optimizer to adjust
    step_size=30,             # Number of epochs between lr reductions
    gamma=0.1,                # Multiplicative factor for lr reduction
    last_epoch=-1,            # Index of last epoch (-1 = auto-initialize)
    verbose='deprecated',     # Deprecated; use get_last_lr() for logging
)

# MultiStepLR — reduce lr at specific milestone epochs
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer=dummy_opt,      # Optimizer to schedule
    milestones=[50, 100],     # List of epoch indices to reduce lr
    gamma=0.1,                # Lr reduction factor at each milestone
    last_epoch=-1,
    verbose='deprecated',
)

# ExponentialLR — multiply lr by gamma every epoch
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer=dummy_opt,      # Optimizer to schedule
    gamma=0.95,               # Multiplicative lr decay applied every epoch
    last_epoch=-1,
    verbose='deprecated',
)

# CosineAnnealingLR — decay lr following cosine curve; good for fine-tuning
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=dummy_opt,      # Optimizer to schedule
    T_max=100,                # Maximum number of iterations (half-cycle length)
    eta_min=0,                # Minimum learning rate reached at cycle trough
    last_epoch=-1,
    verbose='deprecated',
)

# ReduceLROnPlateau — reduce lr when monitored metric stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=dummy_opt,      # Optimizer to schedule
    mode='min',               # 'min' for loss, 'max' for accuracy/metric
    factor=0.1,               # Lr reduction factor when triggered
    patience=10,              # Epochs to wait before reducing lr
    threshold=1e-4,           # Min change to qualify as improvement
    threshold_mode='rel',     # 'rel' relative, 'abs' absolute threshold comparison
    cooldown=0,               # Epochs to wait after reduction before resuming monitoring
    min_lr=0,                 # Minimum lr (floor)
    eps=1e-8,                 # Min lr change to apply reduction
    verbose='deprecated',
)

# OneCycleLR — super-convergence scheduler; used with 1cycle training policy
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer=dummy_opt,      # Optimizer to schedule
    max_lr=0.01,              # Maximum learning rate at cycle peak
    total_steps=None,         # Total number of steps (set either this or epochs*steps_per_epoch)
    epochs=None,              # Number of epochs (used with steps_per_epoch)
    steps_per_epoch=None,     # Steps per epoch (used with epochs)
    pct_start=0.3,            # Fraction of cycle spent increasing lr
    anneal_strategy='cos',    # Annealing: 'cos' cosine, 'linear'
    cycle_momentum=True,      # Cycle momentum inversely with lr
    base_momentum=0.85,       # Lower momentum bound when cycling
    max_momentum=0.95,        # Upper momentum bound when cycling
    div_factor=25.0,          # Initial lr = max_lr / div_factor
    final_div_factor=1e4,     # Final lr = initial_lr / final_div_factor
    three_phase=False,        # If True, use 3 phases (up, down, annihilate)
    last_epoch=-1,
    verbose='deprecated',
)

# WarmupScheduler via LambdaLR — custom warmup function
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer=dummy_opt,      # Optimizer to schedule
    lr_lambda=lambda step: min(step/1000, 1.0),  # Function returning lr multiplier
    last_epoch=-1,
    verbose='deprecated',
)

# ------------------------------------------------------------------------------
# REGULARIZATION / UTILITY LAYERS
# ------------------------------------------------------------------------------

# Flatten — reshapes tensor for transition from conv to linear layers
flatten = nn.Flatten(
    start_dim=1,              # First dimension to flatten (0=batch, 1=start after batch)
    end_dim=-1,               # Last dimension to flatten (-1=last dim)
)

# Unflatten — reverse of Flatten; reshapes flat tensor back to target shape
unflatten = nn.Unflatten(
    dim=1,                    # Dimension to unflatten
    unflattened_size=(8, 8),  # Target shape for the unflattened dimension
)

# Upsample — resize spatial dimensions (decoder / super-resolution)
upsample = nn.Upsample(
    size=None,                # Target output spatial size as int or tuple
    scale_factor=2,           # Multiplier for spatial dimensions
    mode='nearest',           # Interpolation: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    align_corners=None,       # If True, aligns corner pixels (not used for 'nearest')
    recompute_scale_factor=None, # Recompute scale_factor after each forward call
)

# PixelShuffle — efficient sub-pixel upsampling for super-resolution
pixel_shuffle = nn.PixelShuffle(
    upscale_factor=2,         # Factor to increase spatial resolution (rearranges channels → space)
)

# ChannelShuffle — shuffles channels between groups (ShuffleNet)
channel_shuffle = nn.ChannelShuffle(
    groups=4,                 # Number of groups to shuffle channels between
)

# Sequential container — chain layers in order
seq = nn.Sequential(
    nn.Linear(128, 64),       # Each positional arg is a layer executed in order
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 10),
)

# ModuleList — stores modules in a list; does NOT define forward pass order
mod_list = nn.ModuleList(
    modules=[                 # List of modules; iterate manually in forward()
        nn.Linear(64, 64),
        nn.Linear(64, 64),
    ]
)

# ModuleDict — stores modules in a dict; access by name key
mod_dict = nn.ModuleDict(
    modules={                 # Dict of name → module mappings
        'encoder': nn.Linear(128, 64),
        'decoder': nn.Linear(64, 128),
    }
)

# ParameterList — stores nn.Parameter tensors in a list
param_list = nn.ParameterList(
    parameters=[              # List of nn.Parameter objects tracked by the module
        nn.Parameter(torch.randn(64, 64)),
    ]
)

# ParameterDict — stores nn.Parameter tensors in a named dict
param_dict = nn.ParameterDict(
    parameters={              # Dict of name → nn.Parameter
        'weight': nn.Parameter(torch.randn(64, 64)),
    }
)


# ==============================================================================
# SECTION 2 — TORCH.NN.FUNCTIONAL (F) — STATELESS FUNCTIONS
# ==============================================================================

x = torch.randn(8, 128)    # Dummy input

# Activation functions (stateless versions)
out = F.relu(x,
    inplace=False,            # In-place operation flag
)

out = F.leaky_relu(x,
    negative_slope=0.01,      # Slope for negative input region
    inplace=False,
)

out = F.gelu(x,
    approximate='none',       # 'none' for exact; 'tanh' for faster approximation
)

out = F.sigmoid(x)            # Maps values to (0, 1); no parameters

out = F.tanh(x)               # Maps values to (-1, 1); no parameters

out = F.softmax(x,
    dim=1,                    # Dimension to normalize over (class dimension)
    _stacklevel=3,            # Internal stack level for warnings
    dtype=None,               # Override output dtype
)

out = F.log_softmax(x,
    dim=1,                    # Dimension to compute log-softmax over
    _stacklevel=3,
    dtype=None,
)

out = F.dropout(x,
    p=0.5,                    # Probability of zeroing an element
    training=True,            # Apply dropout only when True (disable for inference)
    inplace=False,
)

# Linear operation
weight = torch.randn(64, 128)
bias   = torch.randn(64)
out = F.linear(x,
    weight=weight,            # Weight matrix of shape (out_features, in_features)
    bias=bias,                # Optional bias vector of shape (out_features,)
)

# Normalization functions
out = F.batch_norm(
    input=x.unsqueeze(0).unsqueeze(-1),  # Input tensor
    running_mean=torch.zeros(128),       # Running mean for inference
    running_var=torch.ones(128),         # Running variance for inference
    weight=None,              # Learnable gamma (scale) per feature
    bias=None,                # Learnable beta (shift) per feature
    training=True,            # Use batch stats (True) or running stats (False)
    momentum=0.1,             # Running stats update factor
    eps=1e-5,                 # Numerical stability constant
)

out = F.layer_norm(
    input=x,
    normalized_shape=[128],   # Shape of the features to normalize
    weight=None,              # Learnable scale (gamma) per feature
    bias=None,                # Learnable shift (beta) per feature
    eps=1e-5,                 # Numerical stability constant
)

# Loss functions (functional)
logits = torch.randn(8, 10)
target = torch.randint(0, 10, (8,))

loss = F.cross_entropy(
    input=logits,             # Raw logits of shape (batch, classes)
    target=target,            # Ground truth class indices
    weight=None,              # Per-class weights for imbalance
    size_average=None,        # Deprecated
    ignore_index=-100,        # Index to ignore in loss computation
    reduce=None,              # Deprecated
    reduction='mean',         # 'none', 'mean', 'sum'
    label_smoothing=0.0,      # Label smoothing epsilon
)

loss = F.mse_loss(
    input=torch.randn(8,1),
    target=torch.randn(8,1),
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # 'none', 'mean', 'sum'
)

loss = F.binary_cross_entropy_with_logits(
    input=torch.randn(8,1),
    target=torch.randint(0, 2, (8,1)).float(),
    weight=None,              # Per-sample weights
    size_average=None,        # Deprecated
    reduce=None,              # Deprecated
    reduction='mean',         # Reduction method
    pos_weight=None,          # Weight for positive class (addresses imbalance)
)

# Interpolation / upsampling
feat_map = torch.randn(1, 64, 16, 16)
out = F.interpolate(
    input=feat_map,           # Input tensor to resize
    size=None,                # Target spatial size as int or tuple
    scale_factor=2,           # Multiplier for spatial dimensions
    mode='bilinear',          # Interpolation mode: 'nearest','linear','bilinear','bicubic','trilinear'
    align_corners=False,      # Align corner pixels of input and output grids
    recompute_scale_factor=None, # Recompute scale_factor after call
    antialias=False,          # Apply antialiasing (blurs before downsampling)
)

# Padding
out = F.pad(
    input=x,
    pad=(1, 1, 1, 1),         # Padding: (left, right) for 1D; (left,right,top,bottom) for 2D
    mode='constant',          # 'constant', 'reflect', 'replicate', 'circular'
    value=0,                  # Fill value when mode='constant'
)

# Cosine Similarity
a = torch.randn(8, 128)
b = torch.randn(8, 128)
sim = F.cosine_similarity(
    x1=a,
    x2=b,
    dim=1,                    # Dimension to compute similarity over
    eps=1e-8,                 # Numerical stability constant
)

# Pairwise Distance
dist = F.pairwise_distance(
    x1=a,
    x2=b,
    p=2.0,                    # Norm degree (2=Euclidean, 1=Manhattan)
    eps=1e-6,                 # Numerical stability
    keepdim=False,            # Keep output dimension
)


# ==============================================================================
# SECTION 3 — COMPLETE MODEL ARCHITECTURES (PyTorch)
# ==============================================================================

# ------------------------------------------------------------------------------
# MULTILAYER PERCEPTRON (MLP)
# ------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self,
        input_dim=784,        # Input feature dimension
        hidden_dims=[512,256],# List of hidden layer sizes
        output_dim=10,        # Number of output classes/values
        dropout_p=0.3,        # Dropout probability between layers
        activation=nn.ReLU,   # Activation function class to use
        batch_norm=True,      # Whether to apply batch normalization
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_p))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# CONVOLUTIONAL NEURAL NETWORK (CNN)
# ------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self,
        in_channels=3,        # Input image channels (3=RGB, 1=grayscale)
        num_classes=10,       # Number of output classes
        dropout_p=0.5,        # Dropout probability before classifier
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # Global average pooling → (B, 128, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ------------------------------------------------------------------------------
# AUTOENCODER
# ------------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self,
        input_dim=784,        # Input/output feature dimension
        latent_dim=64,        # Bottleneck (latent space) dimension
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),              # Output in [0,1] for image pixel reconstruction
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ------------------------------------------------------------------------------
# VARIATIONAL AUTOENCODER (VAE)
# ------------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self,
        input_dim=784,        # Input dimension
        hidden_dim=512,       # Hidden layer size
        latent_dim=32,        # Latent space dimension
    ):
        super().__init__()
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)   # Mean of latent distribution
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # Log-variance of latent distribution
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)             # Std from log-variance
        eps = torch.randn_like(std)               # Random noise
        return mu + eps * std                     # Reparameterization trick

    def forward(self, x):
        h = self.encoder_fc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# ------------------------------------------------------------------------------
# LSTM SEQUENCE MODEL
# ------------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self,
        vocab_size=10000,     # Vocabulary size for embedding
        embed_dim=128,        # Embedding vector dimension
        hidden_size=256,      # LSTM hidden state size
        num_layers=2,         # Number of stacked LSTM layers
        num_classes=2,        # Output classes
        dropout=0.3,          # Dropout between LSTM layers
        bidirectional=True,   # Use bidirectional LSTM
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * factor, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (h_n, _) = self.lstm(emb)
        # Use last hidden state (concat forward + backward if bidirectional)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.lstm.bidirectional \
            else h_n[-1]
        return self.classifier(self.dropout(h))

# ------------------------------------------------------------------------------
# TRANSFORMER ENCODER CLASSIFIER
# ------------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self,
        vocab_size=10000,     # Vocabulary size
        embed_dim=256,        # Model embedding dimension
        nhead=8,              # Number of attention heads
        num_layers=4,         # Number of transformer encoder layers
        dim_feedforward=1024, # FFN inner dimension
        num_classes=2,        # Output classes
        max_seq_len=512,      # Maximum input sequence length
        dropout=0.1,          # Dropout probability
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)  # Learned positional embeddings
        enc_layer = nn.TransformerEncoderLayer(
            embed_dim, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)   # Global average pool over sequence
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        emb = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        out = self.transformer(emb)          # (batch, seq, embed_dim)
        pooled = self.pool(out.transpose(1,2)).squeeze(-1)
        return self.classifier(self.dropout(pooled))

# ------------------------------------------------------------------------------
# U-NET (Image Segmentation)
# ------------------------------------------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),   # Conv preserving spatial size
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # Second conv block
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self,
        in_channels=3,        # Input image channels
        out_channels=1,       # Output segmentation mask channels
        features=[64,128,256,512],  # Feature sizes at each encoder stage
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(UNetBlock(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f
        self.bottleneck = UNetBlock(features[-1], features[-1]*2)
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f*2, f, 2, 2))  # Upsample
            self.decoders.append(UNetBlock(f*2, f))                  # Decoder block
        self.head = nn.Conv2d(features[0], out_channels, 1)          # 1×1 final conv

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


# ==============================================================================
# SECTION 4 — TENSORFLOW / KERAS
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, regularizers, callbacks

# ------------------------------------------------------------------------------
# KERAS LAYERS
# ------------------------------------------------------------------------------

# Dense (Fully Connected)
layer = layers.Dense(
    units=128,                # Number of output neurons
    activation='relu',        # Activation: 'relu','sigmoid','tanh','softmax', or callable
    use_bias=True,            # Add learnable bias to each output unit
    kernel_initializer='glorot_uniform',  # Weight init: 'glorot_uniform','he_normal','zeros'
    bias_initializer='zeros', # Bias init strategy
    kernel_regularizer=None,  # Regularizer on weights: e.g. regularizers.L2(1e-4)
    bias_regularizer=None,    # Regularizer on biases
    activity_regularizer=None,# Regularizer applied to layer output
    kernel_constraint=None,   # Constraint on weights (e.g. max-norm)
    bias_constraint=None,     # Constraint on biases
    lora_rank=None,           # LoRA rank for efficient fine-tuning (0=disabled)
    **kwargs,
)

# Conv2D
layer = layers.Conv2D(
    filters=64,               # Number of output feature maps (filters)
    kernel_size=(3, 3),       # Spatial size of the filter: int or (rows, cols)
    strides=(1, 1),           # Stride of convolution: int or (rows, cols)
    padding='same',           # 'valid' no padding, 'same' zero-pad to preserve size
    data_format=None,         # 'channels_last' (NHWC) or 'channels_first' (NCHW)
    dilation_rate=(1, 1),     # Dilation for atrous convolution
    groups=1,                 # Number of input channel groups (depthwise when=in_channels)
    activation=None,          # Activation applied to output
    use_bias=True,            # Add learnable bias
    kernel_initializer='glorot_uniform',  # Filter weight initialization
    bias_initializer='zeros', # Bias initialization
    kernel_regularizer=None,  # Regularizer on filter weights
    bias_regularizer=None,    # Regularizer on biases
    activity_regularizer=None,# Regularizer on layer output
    kernel_constraint=None,   # Constraint on filter weights
    bias_constraint=None,     # Constraint on biases
    **kwargs,
)

# DepthwiseConv2D — applies separate filter per input channel (efficient)
layer = layers.DepthwiseConv2D(
    kernel_size=3,            # Size of depthwise convolutional filter
    strides=(1,1),            # Stride for each spatial dimension
    padding='same',           # Padding mode
    depth_multiplier=1,       # Output channels per input channel
    data_format=None,
    dilation_rate=(1,1),
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',  # Initializer for depthwise filters
    bias_initializer='zeros',
    depthwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    bias_constraint=None,
    **kwargs,
)

# SeparableConv2D — depthwise + pointwise (MobileNet style)
layer = layers.SeparableConv2D(
    filters=128,              # Number of output filters after pointwise step
    kernel_size=3,            # Depthwise kernel size
    strides=(1,1),
    padding='same',
    data_format=None,
    dilation_rate=(1,1),
    depth_multiplier=1,       # Output channels per input channel for depthwise step
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    pointwise_initializer='glorot_uniform',  # Initializer for pointwise (1×1) filters
    bias_initializer='zeros',
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
    **kwargs,
)

# MaxPooling2D
layer = layers.MaxPooling2D(
    pool_size=(2, 2),         # Size of pooling window
    strides=None,             # Step size; defaults to pool_size
    padding='valid',          # 'valid' or 'same'
    data_format=None,         # Channel ordering convention
    name=None,
    **kwargs,
)

# GlobalAveragePooling2D — collapses spatial dims by averaging
layer = layers.GlobalAveragePooling2D(
    data_format=None,         # 'channels_last' or 'channels_first'
    keepdims=False,           # Keep spatial dimensions as size 1 when True
    **kwargs,
)

# LSTM
layer = layers.LSTM(
    units=256,                # Dimensionality of output space (hidden state size)
    activation='tanh',        # Activation for cell state
    recurrent_activation='sigmoid',  # Activation for recurrent step (gates)
    use_bias=True,            # Add bias vectors to gates
    kernel_initializer='glorot_uniform',  # Input weight initializer
    recurrent_initializer='orthogonal',   # Recurrent weight initializer
    bias_initializer='zeros',
    unit_forget_bias=True,    # Initialize forget gate bias to 1 (improves learning)
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,              # Dropout for input connections
    recurrent_dropout=0.0,    # Dropout for recurrent connections (between time steps)
    return_sequences=False,   # Return output for every time step (True) or last only (False)
    return_state=False,       # Return final hidden and cell states
    go_backwards=False,       # Process sequence in reverse order
    stateful=False,           # Reuse final states as initial states for next batch
    unroll=False,             # Unroll loop (faster for short sequences, more memory)
    **kwargs,
)

# GRU
layer = layers.GRU(
    units=128,                # Hidden state / output dimension
    activation='tanh',        # Activation for candidate hidden state
    recurrent_activation='sigmoid',  # Gate activation
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,              # Dropout on input connections
    recurrent_dropout=0.0,    # Dropout on recurrent connections
    return_sequences=False,   # Output at every step vs last step only
    return_state=False,       # Also return final hidden state
    go_backwards=False,       # Reverse input sequence
    stateful=False,           # Persist state across batches
    unroll=False,             # Unroll loop (faster, more memory)
    reset_after=True,         # GRU convention: apply reset gate after matrix multiplication
    **kwargs,
)

# Embedding
layer = layers.Embedding(
    input_dim=10000,          # Vocabulary size (number of unique tokens)
    output_dim=128,           # Embedding vector dimension
    embeddings_initializer='uniform',  # How to initialize embedding weights
    embeddings_regularizer=None,       # Regularizer on embedding weights
    embeddings_constraint=None,        # Constraint on embedding weights
    mask_zero=False,          # Treat 0 as padding and propagate masks
    lora_rank=None,           # LoRA rank for efficient adaptation
    **kwargs,
)

# MultiHeadAttention
layer = layers.MultiHeadAttention(
    num_heads=8,              # Number of parallel attention heads
    key_dim=64,               # Dimension of Q and K per head (total = num_heads * key_dim)
    value_dim=None,           # Dimension of V per head (defaults to key_dim)
    dropout=0.1,              # Dropout on attention weights
    use_bias=True,            # Include bias in projection layers
    output_shape=None,        # Reshape output if set
    attention_axes=None,      # Axes to apply attention over (None=all but batch+last)
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    seed=None,
    **kwargs,
)

# Batch Normalization
layer = layers.BatchNormalization(
    axis=-1,                  # Feature axis to normalize (-1 = last, 1 = channels_first)
    momentum=0.99,            # Running mean/var update momentum
    epsilon=1e-3,             # Numerical stability constant
    center=True,              # Add learnable beta offset
    scale=True,               # Multiply by learnable gamma scale
    beta_initializer='zeros', # Initializer for beta (shift)
    gamma_initializer='ones', # Initializer for gamma (scale)
    moving_mean_initializer='zeros',     # Initial running mean
    moving_variance_initializer='ones',  # Initial running variance
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    synchronized=False,       # Synchronize batch stats across devices (multi-GPU)
    **kwargs,
)

# Layer Normalization
layer = layers.LayerNormalization(
    axis=-1,                  # Axis to normalize over
    epsilon=1e-3,             # Numerical stability constant
    center=True,              # Learn beta (shift)
    scale=True,               # Learn gamma (scale)
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs,
)

# Dropout
layer = layers.Dropout(
    rate=0.5,                 # Fraction of units to randomly drop during training
    noise_shape=None,         # Shape of dropout mask (None=same as input)
    seed=None,                # Random seed for reproducibility
    **kwargs,
)

# Flatten
layer = layers.Flatten(
    data_format=None,         # 'channels_last' or 'channels_first'
    **kwargs,
)

# Reshape
layer = layers.Reshape(
    target_shape=(4, 32),     # Target shape excluding batch dimension
    **kwargs,
)

# Concatenate — merge layers along an axis
layer = layers.Concatenate(
    axis=-1,                  # Axis along which to concatenate tensors
    **kwargs,
)

# Add — element-wise sum of inputs (residual connections)
layer = layers.Add(**kwargs)   # No parameters; sums all input tensors element-wise

# ------------------------------------------------------------------------------
# KERAS LOSSES
# ------------------------------------------------------------------------------

loss = losses.MeanSquaredError(
    reduction='sum_over_batch_size',  # 'sum_over_batch_size','sum','none'
    name='mean_squared_error',
    dtype=None,
)

loss = losses.BinaryCrossentropy(
    from_logits=False,        # True if y_pred is raw logits (no sigmoid applied)
    label_smoothing=0.0,      # Label smoothing factor
    axis=-1,                  # Axis for computing cross-entropy
    reduction='sum_over_batch_size',
    name='binary_crossentropy',
    dtype=None,
)

loss = losses.CategoricalCrossentropy(
    from_logits=False,        # True if predictions are logits
    label_smoothing=0.0,      # Smoothing amount for labels
    axis=-1,                  # Class probability axis
    reduction='sum_over_batch_size',
    name='categorical_crossentropy',
    dtype=None,
)

loss = losses.SparseCategoricalCrossentropy(
    from_logits=False,        # True if predictions are raw logits
    ignore_class=None,        # Class index to ignore (e.g. padding)
    reduction='sum_over_batch_size',
    name='sparse_categorical_crossentropy',
    dtype=None,
)

loss = losses.KLDivergence(
    reduction='sum_over_batch_size',
    name='kl_divergence',
    dtype=None,
)

loss = losses.Huber(
    delta=1.0,                # Threshold between MSE and MAE regions
    reduction='sum_over_batch_size',
    name='huber_loss',
    dtype=None,
)

# ------------------------------------------------------------------------------
# KERAS OPTIMIZERS
# ------------------------------------------------------------------------------

opt = optimizers.Adam(
    learning_rate=1e-3,       # Step size for parameter updates
    weight_decay=None,        # Decoupled weight decay (L2 regularization)
    clipnorm=None,            # Clip gradients by norm threshold
    clipvalue=None,           # Clip each gradient component to [-clipvalue, clipvalue]
    global_clipnorm=None,     # Clip total gradient norm globally
    use_ema=False,            # Use exponential moving average of weights
    ema_momentum=0.99,        # EMA decay factor
    ema_overwrite_frequency=None,  # How often to overwrite weights with EMA
    loss_scale_factor=None,   # Loss scaling factor for mixed precision
    gradient_accumulation_steps=None,  # Accumulate gradients over N steps
    name='adam',
    beta_1=0.9,               # Decay rate for first moment (momentum)
    beta_2=0.999,             # Decay rate for second moment (variance)
    epsilon=1e-7,             # Numerical stability constant
    amsgrad=False,            # Use AMSGrad variant
)

opt = optimizers.SGD(
    learning_rate=0.01,       # Learning rate (step size)
    momentum=0.0,             # Momentum factor
    nesterov=False,           # Use Nesterov accelerated gradient
    weight_decay=None,        # L2 regularization via weight decay
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name='sgd',
)

opt = optimizers.RMSprop(
    learning_rate=0.001,      # Learning rate
    rho=0.9,                  # Decay factor for moving average of squared gradients
    momentum=0.0,             # Momentum factor
    epsilon=1e-7,             # Numerical stability constant
    centered=False,           # Normalize by centered gradient variance when True
    name='rmsprop',
)

opt = optimizers.AdamW(
    learning_rate=1e-3,       # Learning rate
    weight_decay=0.004,       # Decoupled L2 regularization coefficient
    beta_1=0.9,               # First moment decay
    beta_2=0.999,             # Second moment decay
    epsilon=1e-7,             # Stability constant
    amsgrad=False,            # Use AMSGrad
    name='adamw',
)

# ------------------------------------------------------------------------------
# KERAS CALLBACKS
# ------------------------------------------------------------------------------

cb = callbacks.ModelCheckpoint(
    filepath='best_model.keras',        # Path to save model file
    monitor='val_loss',       # Metric to monitor for saving
    verbose=0,                # Verbosity: 0 silent, 1 print on save
    save_best_only=True,      # Only save when monitored metric improves
    save_weights_only=False,  # Save full model (False) or weights only (True)
    mode='auto',              # 'auto','min','max' for monitored metric direction
    save_freq='epoch',        # 'epoch' or integer (save every N batches)
    initial_value_threshold=None,  # Only save if metric better than this baseline
)

cb = callbacks.EarlyStopping(
    monitor='val_loss',       # Metric to monitor
    min_delta=0,              # Min change to qualify as improvement
    patience=10,              # Epochs to wait after last improvement
    verbose=0,                # Verbosity
    mode='auto',              # 'auto','min','max'
    baseline=None,            # Must improve beyond baseline to count
    restore_best_weights=True,# Restore model weights from best epoch
    start_from_epoch=0,       # Epoch to start monitoring from
)

cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss',       # Metric to watch
    factor=0.1,               # Factor by which lr is reduced
    patience=5,               # Epochs with no improvement to wait
    verbose=0,                # Print lr reduction messages
    mode='auto',              # Direction of improvement
    min_delta=1e-4,           # Threshold for measuring improvement
    cooldown=0,               # Epochs to wait after reducing before resuming monitoring
    min_lr=0.0,               # Minimum lr floor
)

cb = callbacks.TensorBoard(
    log_dir='./logs',         # Directory to save log files
    histogram_freq=0,         # Frequency (epochs) to compute weight histograms
    write_graph=True,         # Visualize model graph
    write_images=False,       # Write model weights as images
    write_steps_per_second=False,  # Log steps/second to TensorBoard
    update_freq='epoch',      # 'batch','epoch', or int (every N batches)
    profile_batch=0,          # Batch range to profile for performance
    embeddings_freq=0,        # Frequency to visualize embeddings
    embeddings_metadata=None, # Dict mapping embedding layer → metadata file
)

cb = callbacks.LearningRateScheduler(
    schedule=lambda epoch, lr: lr * 0.95,  # Function (epoch, lr) → new_lr
    verbose=0,                # 1 to print lr at each epoch
)

# ------------------------------------------------------------------------------
# KERAS REGULARIZERS
# ------------------------------------------------------------------------------

reg = regularizers.L1(l1=1e-4)       # L1 penalty coefficient on weights
reg = regularizers.L2(l2=1e-4)       # L2 penalty coefficient on weights
reg = regularizers.L1L2(
    l1=1e-5,                  # L1 penalty strength (drives weights to zero)
    l2=1e-4,                  # L2 penalty strength (shrinks weights)
)

# ------------------------------------------------------------------------------
# KERAS MODEL COMPILATION
# ------------------------------------------------------------------------------

model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',         # Optimizer instance or string name
    loss='sparse_categorical_crossentropy',  # Loss function instance or string
    metrics=['accuracy'],     # List of metrics to track during training
    loss_weights=None,        # Per-output loss weights for multi-output models
    weighted_metrics=None,    # Metrics to evaluate with sample weighting
    run_eagerly=False,        # Run in eager mode for easier debugging (slower)
    steps_per_execution=1,    # Steps fused per tf.function call (GPU efficiency)
    jit_compile='auto',       # XLA JIT compilation: True/False/'auto'
    auto_scale_loss=True,     # Auto loss scaling for mixed precision training
)

model.fit(
    x=None,                   # Input data (numpy array, tf.data.Dataset, etc.)
    y=None,                   # Target labels
    batch_size=32,            # Samples per gradient update
    epochs=10,                # Number of full passes over training data
    verbose='auto',           # 0=silent, 1=progress bar, 2=one line per epoch
    callbacks=None,           # List of callback objects
    validation_split=0.0,     # Fraction of training data for validation
    validation_data=None,     # Explicit (x_val, y_val) validation tuple
    shuffle=True,             # Shuffle training data before each epoch
    class_weight=None,        # Dict {class_id: weight} for imbalanced classes
    sample_weight=None,       # Per-sample weights array
    initial_epoch=0,          # Epoch to start from (useful for resuming)
    steps_per_epoch=None,     # Steps before declaring an epoch done
    validation_steps=None,    # Steps for validation per epoch
    validation_batch_size=None, # Batch size for validation
    validation_freq=1,        # Run validation every N epochs
)
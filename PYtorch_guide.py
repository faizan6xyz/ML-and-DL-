# ==============================================================================
#
#   P Y T O R C H   —   C O M P L E T E   A D V A N C E D   G U I D E
#
#   Covers everything from tensors to distributed training to deployment
#   Every parameter, method, and pattern explained with inline comments
#
#   pip install torch torchvision torchaudio torchtext
#
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch import Tensor
from typing import Optional, List, Tuple, Dict
import numpy as np
import os

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("MPS available:", torch.backends.mps.is_available())   # Apple Silicon GPU


# ==============================================================================
# SECTION 1 — TENSORS: CREATION, TYPES, SHAPES
# ==============================================================================

# ── 1.1  Basic tensor creation ────────────────────────────────────────────────

a = torch.tensor(
    data=[[1.0, 2.0], [3.0, 4.0]],   # Data: list, NumPy array, or scalar
    dtype=torch.float32,               # Data type; inferred if None
    device='cpu',                      # Target device: 'cpu', 'cuda', 'cuda:0', 'mps'
    requires_grad=False,               # Track gradients for autograd (set True for parameters)
    pin_memory=False,                  # Pin in page-locked memory for faster GPU transfer
)

z = torch.zeros(
    *size=(3, 4),                      # Shape as int args or single tuple
    out=None,                          # Optional output tensor to write into
    dtype=None,                        # Default = torch.float32
    layout=torch.strided,              # Memory layout: torch.strided or torch.sparse_coo
    device=None,                       # Target device (None = default)
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,  # Memory layout: contiguous_format, channels_last
)

o = torch.ones(3, 4,                   # Shape passed as positional args
    dtype=torch.float32,
    device=None,
    requires_grad=False,
)

e = torch.empty(
    *size=(3, 4),                      # Uninitialized tensor (contents are garbage)
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
)

f = torch.full(
    size=(3, 3),                       # Output shape
    fill_value=7.0,                    # Value to fill all elements with
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)

eye = torch.eye(
    n=4,                               # Number of rows
    m=None,                            # Number of columns (square if None)
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)

r = torch.rand(
    *size=(4, 4),                      # Shape: uniform random in [0, 1)
    generator=None,                    # Optional torch.Generator for reproducibility
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
)

rn = torch.randn(
    *size=(4, 4),                      # Shape: standard normal distribution N(0, 1)
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
)

ri = torch.randint(
    low=0,                             # Lower bound (inclusive)
    high=10,                           # Upper bound (exclusive)
    size=(4, 4),                       # Output shape
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)

rng = torch.arange(
    start=0,                           # Start value (or stop if only 1 arg)
    end=10,                            # Stop value (exclusive)
    step=1,                            # Step between values
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)

lin = torch.linspace(
    start=0.0,                         # Start value (inclusive)
    end=1.0,                           # End value (inclusive)
    steps=11,                          # Number of evenly spaced values
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)

# From NumPy (shares memory — changes to one affect the other!)
np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t_from_np = torch.from_numpy(np_arr)          # Zero-copy: shares memory with NumPy array
t_copy    = torch.tensor(np_arr)               # Copy: independent from NumPy array

# ── 1.2  Tensor properties ────────────────────────────────────────────────────

t = torch.randn(2, 3, 4)
print(t.shape)                 # torch.Size([2, 3, 4]) — tuple-like shape
print(t.size())                # Alias for shape
print(t.dtype)                 # Data type (torch.float32, etc.)
print(t.device)                # Device (cpu / cuda:0)
print(t.ndim)                  # Number of dimensions
print(t.numel())               # Total number of elements
print(t.is_cuda)               # True if on GPU
print(t.is_contiguous())       # True if stored contiguously in memory
print(t.stride())              # Strides tuple (bytes step per dimension)
print(t.storage())             # Underlying storage object
print(t.requires_grad)         # Whether gradients are tracked
print(t.grad)                  # Accumulated gradient (None if no backward yet)
print(t.grad_fn)               # Function that created this tensor (for autograd graph)

# ── 1.3  Type casting & device transfer ──────────────────────────────────────

t_int   = torch.tensor([1, 2, 3])
t_fp32  = t_int.float()                        # Cast to float32
t_fp16  = t_fp32.half()                        # Cast to float16 (fp16)
t_bf16  = t_fp32.bfloat16()                    # Cast to bfloat16 (better range than fp16)
t_fp64  = t_fp32.double()                      # Cast to float64
t_int32 = t_fp32.int()                         # Cast to int32
t_int64 = t_fp32.long()                        # Cast to int64 (most common for indices)
t_bool  = t_int.bool()                         # Cast to bool

t_cast = t_fp32.to(
    dtype=torch.float16,       # Target dtype
    device='cuda',             # Target device (can also pass device='cpu')
    non_blocking=False,        # Async transfer (True = overlap with compute, needs pin_memory)
    copy=False,                # Force copy even if dtype/device already match
    memory_format=torch.preserve_format,  # Memory layout after cast
)

# Device transfer
t_gpu = t.cuda(
    device=0,                  # GPU index (int or 'cuda:0')
    non_blocking=False,        # Async GPU transfer
)
t_cpu = t_gpu.cpu()            # Move to CPU
t_mps = t.to('mps')           # Move to Apple Silicon GPU

# ── 1.4  Shape manipulation ───────────────────────────────────────────────────

t = torch.randn(2, 3, 4)

reshaped = t.reshape(6, 4)     # Reshape (may share data if contiguous)
viewed   = t.view(6, 4)        # View (must be contiguous; shares data)
t.contiguous()                 # Return contiguous copy if not already contiguous

transposed = t.transpose(
    dim0=1,                    # First dimension to swap
    dim1=2,                    # Second dimension to swap
)
permuted = t.permute(0, 2, 1)  # Permute all dims at once (like np.transpose)

squeezed = t.squeeze(
    dim=None,                  # Dimension to squeeze (None removes all size-1 dims)
)
expanded = t.unsqueeze(
    dim=0,                     # Insert a new size-1 dimension at this position
)

repeated = t.repeat(
    *repeats=(2, 1, 3),        # Repeat along each dim this many times
)
expanded2 = t.expand(
    *sizes=(4, 3, 4),          # Target sizes (-1 = keep existing); shares memory
)

flattened = t.flatten(
    start_dim=0,               # First dim to flatten
    end_dim=-1,                # Last dim to flatten (-1 = last)
)

# Stack and cat
a = torch.randn(2, 3)
b = torch.randn(2, 3)
stacked = torch.stack(
    tensors=[a, b],            # Sequence of tensors to stack
    dim=0,                     # Dimension to insert for new axis
    out=None,
)
catted = torch.cat(
    tensors=[a, b],            # Sequence of tensors to concatenate
    dim=0,                     # Existing dimension to concatenate along
    out=None,
)
chunks = torch.chunk(
    input=a,                   # Input tensor to split
    chunks=2,                  # Number of chunks
    dim=0,                     # Dimension to split along
)
splits = torch.split(
    tensor=a,                  # Input tensor
    split_size_or_sections=1,  # Int (equal splits) or list of sizes
    dim=0,                     # Dimension to split along
)

# ── 1.5  Indexing and slicing ─────────────────────────────────────────────────

t = torch.randn(4, 5, 6)

# Standard indexing (Python-style)
t[0]                           # First element along dim 0
t[1:3, ::2]                    # Slicing along multiple dims
t[[0, 2], :]                   # Fancy indexing with list of indices

# torch.index_select
selected = torch.index_select(
    input=t,                   # Input tensor
    dim=0,                     # Dimension to select along
    index=torch.tensor([0, 2]),# 1-D tensor of indices to select
    out=None,
)

# torch.gather — for per-row or per-element selection
idx = torch.randint(0, 5, (4, 3))
gathered = torch.gather(
    input=torch.randn(4, 5),   # Input tensor
    dim=1,                     # Dimension to gather along
    index=idx,                 # Index tensor (same ndim as input)
    sparse_grad=False,         # Use sparse gradient for input
    out=None,
)

# Boolean masking
mask    = t > 0
masked  = t[mask]              # Returns 1-D tensor of elements where mask is True
t[mask] = 0                    # In-place masked assignment

# torch.where
result = torch.where(
    condition=t > 0,           # Boolean condition tensor
    input=t,                   # Values where condition is True
    other=torch.zeros_like(t), # Values where condition is False
)

# nonzero indices
indices = torch.nonzero(
    input=t > 0,               # Input tensor
    out=None,
    as_tuple=False,            # True returns tuple of 1-D tensors; False returns N×ndim matrix
)

# ── 1.6  Math operations ─────────────────────────────────────────────────────

a = torch.randn(3, 3)
b = torch.randn(3, 3)

# Element-wise
torch.add(a, b)                # a + b
torch.sub(a, b)                # a - b
torch.mul(a, b)                # a * b (Hadamard)
torch.div(a, b)                # a / b
torch.pow(a, 2)                # a ** 2
torch.sqrt(torch.abs(a))       # sqrt of absolute values
torch.abs(a)                   # |a|
torch.exp(a)                   # e^a
torch.log(torch.abs(a) + 1e-6) # ln(a) — add eps for numerical safety
torch.log2(torch.abs(a) + 1)
torch.log10(torch.abs(a) + 1)
torch.sign(a)                  # -1, 0, or +1
torch.floor(a)                 # Floor
torch.ceil(a)                  # Ceiling
torch.round(a)                 # Round to nearest integer
torch.clamp(a,
    min=-1.0,                  # Lower bound (floor clamp)
    max=1.0,                   # Upper bound (ceiling clamp)
    out=None,
)

# Matrix operations
torch.mm(a, b)                 # Matrix multiply (2D only)
torch.matmul(a, b)             # Matrix multiply (broadcasts over batch dims)
a @ b                          # Same as torch.matmul
torch.bmm(                     # Batch matrix multiply (3D tensors only)
    input=torch.randn(8, 3, 4),
    mat2=torch.randn(8, 4, 5),
    out=None,
)
torch.linalg.inv(a)            # Matrix inverse
torch.linalg.det(a)            # Determinant
torch.linalg.eig(a)            # Eigenvalues and eigenvectors (complex output)
torch.linalg.eigh(a)           # Eigendecomposition of symmetric/Hermitian matrix
U, S, Vh = torch.linalg.svd(a, full_matrices=True)  # SVD decomposition
torch.linalg.norm(a,
    ord=None,                  # Norm type: None (Frobenius), 1, 2, 'fro', 'nuc', inf
    dim=None,                  # Dimension(s) to compute norm over
    keepdim=False,
    dtype=None,
    out=None,
)
torch.linalg.solve(a, b)       # Solve linear system Ax = b

# Reductions
torch.sum(a, dim=1, keepdim=False, dtype=None, out=None)
torch.mean(a, dim=0, keepdim=False, dtype=None, out=None)
torch.max(a)                   # Global max (returns tensor)
torch.max(a, dim=1)            # Per-row max: returns (values, indices)
torch.min(a, dim=0)            # Per-column min
torch.prod(a, dim=1)           # Product along dim
torch.std(a, dim=0,
    correction=1,              # Bessel's correction: 1=unbiased, 0=biased
    keepdim=False,
    out=None,
)
torch.var(a, dim=0, correction=1, keepdim=False, out=None)
torch.cumsum(a, dim=0)         # Cumulative sum along dim
torch.cumprod(a, dim=0)        # Cumulative product along dim
torch.median(a)                # Median (returns tensor for global, namedtuple for dim)
torch.mode(a, dim=0)           # Most frequent value and index
torch.argmax(a, dim=1, keepdim=False)  # Index of maximum value along dim
torch.argmin(a, dim=0)
vals, idxs = torch.topk(
    input=a.flatten(),
    k=3,                       # Number of top elements
    dim=-1,                    # Dimension to select from
    largest=True,              # True=largest, False=smallest
    sorted=True,               # Return sorted results
    out=None,
)

# Sorting
sorted_t, indices = torch.sort(
    input=a,
    dim=-1,                    # Dimension to sort along
    descending=False,          # Sort in descending order
    stable=False,              # Stable sort (preserve relative order of equal elements)
    out=None,
)
torch.argsort(a, dim=0, descending=False, stable=False)

# Comparison
torch.eq(a, b)                 # Element-wise equality
torch.ne(a, b)                 # Not equal
torch.gt(a, b)                 # Greater than
torch.lt(a, b)                 # Less than
torch.ge(a, b)                 # Greater than or equal
torch.le(a, b)                 # Less than or equal
torch.isnan(a)                 # True where NaN
torch.isinf(a)                 # True where infinite
torch.isfinite(a)              # True where finite
torch.allclose(a, b,
    rtol=1e-5,                 # Relative tolerance
    atol=1e-8,                 # Absolute tolerance
    equal_nan=False,           # Treat NaN as equal if True
)


# ==============================================================================
# SECTION 2 — AUTOGRAD: AUTOMATIC DIFFERENTIATION
# ==============================================================================

# ── 2.1  Basic gradient computation ─────────────────────────────────────────

x = torch.tensor(3.0, requires_grad=True)  # Leaf tensor with grad tracking
y = x ** 2 + 3 * x + 1

y.backward(
    gradient=None,             # External gradient (for non-scalar y, must provide)
    retain_graph=False,        # Keep computation graph after backward (for multiple calls)
    create_graph=False,        # Create graph of derivatives (for higher-order grads)
    inputs=None,               # Accumulate gradients only for these inputs
)
print(x.grad)                  # dy/dx = 2x + 3 = 9 at x=3

# Manual gradient zeroing (required before each backward)
x.grad.zero_()                 # Zero gradient in-place (must do before next backward)

# ── 2.2  Gradient context managers ───────────────────────────────────────────

# Disable gradient tracking (inference mode)
with torch.no_grad():
    y = x * 2                  # No gradient graph built (faster, less memory)

# Enable gradients inside a no_grad context
with torch.no_grad():
    with torch.enable_grad():
        y = x ** 2             # Gradients enabled again inside

# Inference mode (stronger than no_grad — tensors can't be used in autograd)
with torch.inference_mode(mode=True):
    y = x + 1                  # Fastest inference option; no grad graph at all

# Temporarily toggle requires_grad
x = torch.randn(3, requires_grad=True)
x.requires_grad_(False)        # Detach in-place
x.requires_grad_(True)         # Re-enable in-place

# Detach from graph (returns new tensor sharing data but without grad history)
y = x.detach()                 # Detached tensor; safe to use as NumPy/constant

# ── 2.3  torch.autograd.grad — explicit gradient computation ─────────────────

x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()

grads = torch.autograd.grad(
    outputs=y,                 # Scalar or tuple of outputs
    inputs=x,                  # Input(s) to differentiate w.r.t.
    grad_outputs=None,         # Upstream gradients (for vector-valued outputs)
    retain_graph=False,        # Keep graph for multiple calls
    create_graph=False,        # Build graph through gradient (for higher-order)
    only_inputs=True,          # Only compute grads for specified inputs
    allow_unused=False,        # Allow inputs not contributing to output
    is_grads_batched=False,    # Interpret grad_outputs as batched
    materialize_grads=False,   # Return zero grad instead of None for unused
)

# Higher-order gradients
x = torch.randn(1, requires_grad=True)
y = x ** 3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]     # Second derivative

# ── 2.4  Custom autograd Function ────────────────────────────────────────────

class SoftClamp(torch.autograd.Function):
    """Custom activation with defined forward and backward passes."""

    @staticmethod
    def forward(
        ctx,                   # Context object to stash info for backward
        input,                 # Input tensor
        alpha=1.0,             # Custom parameter
    ):
        ctx.save_for_backward(input)            # Save tensors needed in backward
        ctx.alpha = alpha                       # Save non-tensor state
        return torch.clamp(input, -alpha, alpha)

    @staticmethod
    def backward(
        ctx,                   # Context with saved tensors
        grad_output,           # Upstream gradient
    ):
        input, = ctx.saved_tensors             # Retrieve saved tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() >= ctx.alpha] = 0  # Zero grad outside clamp
        return grad_input, None                 # One grad per forward arg (None for alpha)

# Apply custom function
soft_clamp = SoftClamp.apply
output = soft_clamp(torch.randn(4, requires_grad=True), 0.5)


# ==============================================================================
# SECTION 3 — nn.MODULE: LAYERS AND MODELS
# ==============================================================================

# ── 3.1  All core nn layers ──────────────────────────────────────────────────

# LINEAR
nn.Linear(
    in_features=128,           # Input feature size
    out_features=64,           # Output feature size
    bias=True,                 # Add learnable bias
    device=None,               # Device for parameters
    dtype=None,                # Dtype for parameters
)

nn.Bilinear(
    in1_features=64,           # First input feature size
    in2_features=64,           # Second input feature size
    out_features=32,           # Output feature size
    bias=True,                 # Include bias
    device=None,
    dtype=None,
)

# CONVOLUTIONS
nn.Conv1d(
    in_channels=1,             # Input channels
    out_channels=32,           # Number of filters
    kernel_size=3,             # Filter length
    stride=1,                  # Step size
    padding=0,                 # Zero-padding on each side
    dilation=1,                # Spacing between kernel elements
    groups=1,                  # Grouped convolution (depthwise if == in_channels)
    bias=True,
    padding_mode='zeros',      # 'zeros','reflect','replicate','circular'
    device=None,
    dtype=None,
)

nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,             # Int or (H, W) tuple
    stride=1,                  # Int or (H, W) tuple
    padding=1,                 # Int, (H,W), or 'same'/'valid'
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    device=None,
    dtype=None,
)

nn.Conv3d(
    in_channels=1,
    out_channels=16,
    kernel_size=3,             # Int or (D, H, W)
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros',
    device=None,
    dtype=None,
)

nn.ConvTranspose2d(
    in_channels=64,
    out_channels=32,
    kernel_size=4,
    stride=2,                  # Upsampling factor (stride > 1 increases spatial size)
    padding=1,
    output_padding=0,          # Extra rows/cols added to output one side
    groups=1,
    bias=True,
    dilation=1,
    padding_mode='zeros',
    device=None,
    dtype=None,
)

# POOLING
nn.MaxPool1d(kernel_size=2, stride=None, padding=0, dilation=1,
             return_indices=False, ceil_mode=False)
nn.MaxPool2d(
    kernel_size=2,             # Pooling window size
    stride=None,               # Step size (defaults to kernel_size)
    padding=0,
    dilation=1,
    return_indices=False,      # Return argmax indices (for MaxUnpool2d)
    ceil_mode=False,           # Use ceil instead of floor for output size
)
nn.MaxPool3d(kernel_size=2, stride=None, padding=0, dilation=1,
             return_indices=False, ceil_mode=False)
nn.AvgPool2d(
    kernel_size=2,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,    # Include zero-padding in average calculation
    divisor_override=None,     # Custom divisor instead of window size
)
nn.AdaptiveAvgPool1d(output_size=1)          # Output size regardless of input size
nn.AdaptiveAvgPool2d(output_size=(1, 1))     # (1,1) → Global Average Pooling
nn.AdaptiveMaxPool2d(output_size=7, return_indices=False)
nn.LPPool2d(norm_type=2, kernel_size=2, stride=None, ceil_mode=False)  # Lp pooling

# NORMALIZATION
nn.BatchNorm1d(
    num_features=128,          # Number of features/channels
    eps=1e-5,                  # Numerical stability constant
    momentum=0.1,              # Running mean/var update rate
    affine=True,               # Learn scale (gamma) and shift (beta)
    track_running_stats=True,  # Track running stats for inference
    device=None,
    dtype=None,
)
nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1,
               affine=True, track_running_stats=True)
nn.BatchNorm3d(num_features=32, eps=1e-5, momentum=0.1,
               affine=True, track_running_stats=True)

nn.LayerNorm(
    normalized_shape=512,      # Shape to normalize; int or list
    eps=1e-5,
    elementwise_affine=True,   # Learn per-element gamma and beta
    bias=True,                 # Include bias (beta) term
    device=None,
    dtype=None,
)

nn.GroupNorm(
    num_groups=8,              # Number of channel groups
    num_channels=64,           # Total channels (must be divisible by num_groups)
    eps=1e-5,
    affine=True,               # Learn per-channel scale and shift
    device=None,
    dtype=None,
)

nn.InstanceNorm2d(
    num_features=64,           # Number of channels
    eps=1e-5,
    momentum=0.1,
    affine=False,              # Usually False (no learned params) for style transfer
    track_running_stats=False, # Usually False for instance norm
    device=None,
    dtype=None,
)

nn.RMSNorm(
    normalized_shape=512,      # Shape to normalize (Llama/GPT-style normalization)
    eps=1e-5,
    elementwise_affine=True,   # Learn per-element scale (no bias in RMSNorm)
    device=None,
    dtype=None,
)

# ACTIVATIONS
nn.ReLU(inplace=False)         # max(0, x); inplace saves memory but breaks autograd in some cases
nn.ReLU6(inplace=False)        # min(max(0, x), 6); used in MobileNet
nn.LeakyReLU(negative_slope=0.01, inplace=False)  # Allows small gradient when x < 0
nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)  # Learned slope
nn.ELU(alpha=1.0, inplace=False)         # Smooth negative saturation
nn.SELU(inplace=False)                   # Self-normalizing; use with AlphaDropout
nn.GELU(approximate='none')              # 'none'=exact, 'tanh'=approx (used in BERT/GPT)
nn.SiLU(inplace=False)                   # Swish: x * sigmoid(x); used in EfficientNet
nn.Mish(inplace=False)                   # x * tanh(softplus(x)); smooth self-gated
nn.Sigmoid()                             # 1/(1+e^-x); squashes to (0,1)
nn.Tanh()                                # (e^x-e^-x)/(e^x+e^-x); squashes to (-1,1)
nn.Softmax(dim=1)                        # Normalizes to sum=1 along dim
nn.LogSoftmax(dim=1)                     # log(softmax); more numerically stable for NLLLoss
nn.Softplus(beta=1, threshold=20)        # Smooth ReLU approximation
nn.Softsign()                            # x/(1+|x|)
nn.Hardshrink(lambd=0.5)                 # Set small values to 0
nn.Hardswish(inplace=False)              # Efficient Swish for mobile (MobileNetV3)
nn.Hardsigmoid(inplace=False)            # Piecewise linear sigmoid approximation
nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)
nn.Threshold(threshold=0.5, value=0, inplace=False)  # Set values below threshold to value
nn.MultiheadAttention                    # See attention section below

# DROPOUT
nn.Dropout(
    p=0.5,                     # Probability of zeroing each element
    inplace=False,             # Modify input in-place
)
nn.Dropout1d(p=0.5, inplace=False)       # Zero entire 1D feature channels
nn.Dropout2d(p=0.5, inplace=False)       # Zero entire 2D feature maps
nn.Dropout3d(p=0.5, inplace=False)       # Zero entire 3D feature volumes
nn.AlphaDropout(p=0.5, inplace=False)    # Dropout preserving mean/var (use with SELU)

# RECURRENT
nn.RNN(
    input_size=128,            # Features per time step
    hidden_size=256,           # Hidden state dimension
    num_layers=1,              # Stacked layers
    nonlinearity='tanh',       # 'tanh' or 'relu'
    bias=True,
    batch_first=False,         # (batch, seq, feature) when True
    dropout=0.0,               # Dropout between layers (not last layer)
    bidirectional=False,       # Process in both directions
    device=None,
    dtype=None,
)

nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    bias=True,
    batch_first=True,          # (batch, seq, feature) input order
    dropout=0.3,               # Dropout between layers (0 if num_layers=1)
    bidirectional=False,       # Bidirectional doubles output features
    proj_size=0,               # LSTMP projection size (0 = disabled)
    device=None,
    dtype=None,
)

nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=1,
    bias=True,
    batch_first=True,
    dropout=0.0,
    bidirectional=False,
    device=None,
    dtype=None,
)

# EMBEDDING
nn.Embedding(
    num_embeddings=10000,      # Vocabulary size
    embedding_dim=256,         # Embedding vector dimension
    padding_idx=0,             # Index whose embedding stays zero (padding token)
    max_norm=None,             # Normalize embeddings with norm > max_norm
    norm_type=2.0,             # Norm type for max_norm
    scale_grad_by_freq=False,  # Scale gradients by inverse token frequency
    sparse=False,              # Use sparse gradient updates (faster for large vocab)
    _weight=None,              # Pre-loaded embedding weights
    _freeze=False,             # Freeze weights (no gradient updates)
    device=None,
    dtype=None,
)

nn.EmbeddingBag(
    num_embeddings=10000,
    embedding_dim=128,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    mode='mean',               # Aggregation: 'mean', 'sum', 'max'
    sparse=False,
    _weight=None,
    include_last_offset=False, # Include final offset as end-of-bag marker
    padding_idx=None,
    device=None,
    dtype=None,
)

# TRANSFORMER LAYERS
nn.MultiheadAttention(
    embed_dim=512,             # Total embedding dimension (Q, K, V projections)
    num_heads=8,               # Number of attention heads (embed_dim % num_heads == 0)
    dropout=0.1,               # Dropout on attention weights
    bias=True,                 # Add bias to projections
    add_bias_kv=False,         # Add bias to key and value sequences
    add_zero_attn=False,       # Append a batch of zeros to K and V
    kdim=None,                 # Key projection dimension (defaults to embed_dim)
    vdim=None,                 # Value projection dimension (defaults to embed_dim)
    batch_first=True,          # (batch, seq, feature) when True
    device=None,
    dtype=None,
)

nn.TransformerEncoderLayer(
    d_model=512,               # Input/output model dimension
    nhead=8,                   # Number of attention heads
    dim_feedforward=2048,      # FFN hidden dimension (usually 4× d_model)
    dropout=0.1,               # Dropout throughout the layer
    activation='relu',         # FFN activation: 'relu', 'gelu', or callable
    layer_norm_eps=1e-5,
    batch_first=True,          # (batch, seq, feature) input
    norm_first=False,          # Pre-norm (True) vs post-norm (False)
    bias=True,                 # Use bias in all linear layers
    device=None,
    dtype=None,
)

nn.TransformerDecoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-5,
    batch_first=True,
    norm_first=False,
    bias=True,
    device=None,
    dtype=None,
)

nn.TransformerEncoder(
    encoder_layer=nn.TransformerEncoderLayer(512, 8, batch_first=True),
    num_layers=6,              # Number of encoder layers to stack
    norm=None,                 # Optional final normalization layer
    enable_nested_tensor=True, # Use nested tensors for padding efficiency
    mask_check=True,           # Validate masks at forward call
)

nn.TransformerDecoder(
    decoder_layer=nn.TransformerDecoderLayer(512, 8, batch_first=True),
    num_layers=6,
    norm=None,
)

nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu',
    custom_encoder=None,       # Override default encoder
    custom_decoder=None,       # Override default decoder
    layer_norm_eps=1e-5,
    batch_first=True,
    norm_first=False,
    bias=True,
    device=None,
    dtype=None,
)

# UTILITY LAYERS
nn.Flatten(start_dim=1, end_dim=-1)       # Flatten dims [start_dim, end_dim]
nn.Unflatten(dim=1, unflattened_size=(8, 8))  # Reverse of Flatten
nn.Identity(*args, **kwargs)               # Pass-through (placeholder/testing)
nn.PixelShuffle(upscale_factor=2)          # Rearrange channels to spatial (super-res)
nn.PixelUnshuffle(downscale_factor=2)      # Inverse of PixelShuffle
nn.Upsample(size=None, scale_factor=2,
            mode='nearest',               # 'nearest','linear','bilinear','bicubic','trilinear'
            align_corners=None,
            recompute_scale_factor=None)
nn.UpsamplingBilinear2d(size=None, scale_factor=2)   # Shortcut for bilinear Upsample
nn.ChannelShuffle(groups=4)               # Shuffle channels between groups (ShuffleNet)
nn.Fold(output_size=(8,8), kernel_size=3, stride=1, padding=0, dilation=1)
nn.Unfold(kernel_size=3, dilation=1, padding=0, stride=1)  # Extract sliding patches

# LOSS FUNCTIONS
nn.MSELoss(
    size_average=None,         # Deprecated
    reduce=None,               # Deprecated
    reduction='mean',          # 'none', 'mean', 'sum'
)
nn.L1Loss(reduction='mean')
nn.HuberLoss(reduction='mean', delta=1.0)   # MSE for small, MAE for large errors
nn.SmoothL1Loss(reduction='mean', beta=1.0) # Similar to Huber; used in detection

nn.BCELoss(
    weight=None,               # Per-sample weights
    size_average=None,
    reduce=None,
    reduction='mean',
)
nn.BCEWithLogitsLoss(
    weight=None,               # Per-sample weights
    size_average=None,
    reduce=None,
    reduction='mean',
    pos_weight=None,           # Weight for positive class (> 1 boosts recall)
)
nn.CrossEntropyLoss(
    weight=None,               # Per-class weights for class imbalance
    size_average=None,
    ignore_index=-100,         # Class index to ignore in loss (e.g. padding)
    reduce=None,
    reduction='mean',
    label_smoothing=0.0,       # Label smoothing epsilon (0.1 is common)
)
nn.NLLLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean',
)
nn.KLDivLoss(
    size_average=None,
    reduce=None,
    reduction='mean',          # Use 'batchmean' for mathematically correct KL
    log_target=False,          # If True, target is already in log-space
)
nn.CTCLoss(
    blank=0,                   # Blank label index used in CTC
    reduction='mean',
    zero_infinity=False,       # Set infinite losses and their gradients to zero
)
nn.TripletMarginLoss(
    margin=1.0,                # Margin between positive and negative distances
    p=2,                       # Norm degree for distance
    eps=1e-6,
    swap=False,                # Use distance swap for semi-hard mining
    size_average=None,
    reduce=None,
    reduction='mean',
)
nn.CosineEmbeddingLoss(
    margin=0.0,                # Margin for negative pairs
    size_average=None,
    reduce=None,
    reduction='mean',
)
nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='mean')
nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None,
                   reduce=None, reduction='mean')
nn.PoissonNLLLoss(log_input=True, full=False, size_average=None,
                  eps=1e-8, reduce=None, reduction='mean')
nn.GaussianNLLLoss(full=False, eps=1e-6, reduction='mean')  # Heteroscedastic regression

# ── 3.2  Container modules ────────────────────────────────────────────────────

# Sequential — ordered chain of layers
seq = nn.Sequential(
    nn.Linear(128, 64),        # Layers executed in positional order
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 10),
)
# Or with OrderedDict for named access
from collections import OrderedDict
seq = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(128, 64)),
    ('act', nn.ReLU()),
    ('fc2', nn.Linear(64, 10)),
]))
seq.fc1                        # Access layer by name

# ModuleList — list of modules with proper parameter tracking
mod_list = nn.ModuleList(
    modules=[nn.Linear(64, 64), nn.Linear(64, 64)]  # Iterable in forward manually
)

# ModuleDict — named dict of modules with parameter tracking
mod_dict = nn.ModuleDict(
    modules={'encoder': nn.Linear(128, 64), 'decoder': nn.Linear(64, 128)}
)

# ParameterList / ParameterDict — tracked raw parameters
param_list = nn.ParameterList([nn.Parameter(torch.randn(64))])
param_dict = nn.ParameterDict({'scale': nn.Parameter(torch.ones(1))})


# ==============================================================================
# SECTION 4 — BUILDING CUSTOM MODELS
# ==============================================================================

# ── 4.1  Base nn.Module pattern ───────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-activation residual block (He et al.)."""

    def __init__(self,
        in_channels: int,      # Input channel count
        out_channels: int,     # Output channel count
        stride: int = 1,       # Stride for downsampling
        dropout_p: float = 0.0,# Dropout probability
    ):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.act   = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.drop  = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        # Projection shortcut if shape changes
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.act(self.bn1(x)))  # Pre-activation
        out = self.conv2(self.act(self.bn2(out)))
        out = self.drop(out)
        return out + self.shortcut(x)             # Residual addition


class ResNet(nn.Module):
    """Simple ResNet built from residual blocks."""

    def __init__(self,
        block=ResidualBlock,
        layers=[2, 2, 2, 2],   # Blocks per stage
        num_classes=1000,
        in_channels=3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64,  64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64,  128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # Global average pooling
        self.fc      = nn.Linear(512, num_classes)

        # Weight initialization (critical for convergence)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)    # Initialize gamma = 1
                nn.init.constant_(m.bias, 0)      # Initialize beta = 0
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride):
        layers = [block(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    # Useful model utilities
    def freeze_backbone(self):
        """Freeze all layers except final classifier."""
        for name, param in self.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


# ── 4.2  Weight initialization ───────────────────────────────────────────────

def init_weights(module: nn.Module):
    """Apply standard weight initialization to a model."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)   # Glorot uniform: good for tanh/sigmoid
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(               # He normal: optimal for ReLU networks
            module.weight,
            a=0,                               # Leakiness (0 = standard ReLU)
            mode='fan_out',                    # 'fan_in' or 'fan_out' for mode
            nonlinearity='relu',               # Nonlinearity following this layer
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(module.weight)           # gamma = 1
        nn.init.zeros_(module.bias)            # beta = 0

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)   # GPT-style init
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])  # Zero padding vector

# All init functions:
# nn.init.uniform_(tensor, a=0.0, b=1.0)
# nn.init.normal_(tensor, mean=0.0, std=1.0)
# nn.init.constant_(tensor, val)
# nn.init.ones_(tensor)
# nn.init.zeros_(tensor)
# nn.init.eye_(tensor)                      # Identity matrix (2D only)
# nn.init.xavier_uniform_(tensor, gain=1.0)
# nn.init.xavier_normal_(tensor, gain=1.0)
# nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
# nn.init.kaiming_normal_(tensor, a=0, mode='fan_out', nonlinearity='relu')
# nn.init.orthogonal_(tensor, gain=1.0)     # Orthogonal matrix (good for RNNs)
# nn.init.sparse_(tensor, sparsity, std=0.01)
# nn.init.trunc_normal_(tensor, mean=0, std=1, a=-2, b=2)  # Truncated normal


# ==============================================================================
# SECTION 5 — OPTIMIZERS
# ==============================================================================

model = ResNet()
params = model.parameters()

# ── 5.1  All optimizers ───────────────────────────────────────────────────────

optim.SGD(
    params=model.parameters(), # Iterable of parameters or list of param groups
    lr=0.01,                   # Learning rate (required)
    momentum=0.9,              # Momentum factor (0 = pure GD)
    dampening=0,               # Dampening for momentum
    weight_decay=1e-4,         # L2 regularization coefficient
    nesterov=True,             # Nesterov accelerated gradient (look-ahead step)
    maximize=False,            # Maximize objective instead of minimize
    foreach=None,              # Use vectorized foreach implementation (faster on CUDA)
    differentiable=False,      # Differentiate through optimizer step
)

optim.Adam(
    params=model.parameters(),
    lr=1e-3,                   # Learning rate
    betas=(0.9, 0.999),        # (beta1, beta2): decay rates for moment estimates
    eps=1e-8,                  # Numerical stability constant in denominator
    weight_decay=0,            # L2 regularization (adds to gradient)
    amsgrad=False,             # AMSGrad: use max of past squared grads for stability
    maximize=False,
    foreach=None,
    capturable=False,          # Allow use in CUDA graph capture
    differentiable=False,
    fused=None,                # Fused CUDA kernel (fastest on GPU, requires CUDA)
)

optim.AdamW(
    params=model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,         # Decoupled weight decay (not added to gradient)
    amsgrad=False,
    maximize=False,
    foreach=None,
    capturable=False,
    differentiable=False,
    fused=None,
)

optim.RMSprop(
    params=model.parameters(),
    lr=0.01,
    alpha=0.99,                # Smoothing constant for squared gradient average
    eps=1e-8,
    weight_decay=0,
    momentum=0,
    centered=False,            # Normalize by centered gradient variance if True
    maximize=False,
    foreach=None,
    differentiable=False,
    capturable=False,
)

optim.Adagrad(
    params=model.parameters(),
    lr=0.01,
    lr_decay=0,                # Learning rate decay across updates
    weight_decay=0,
    initial_accumulator_value=0,  # Starting value for gradient accumulator
    eps=1e-10,
    maximize=False,
    foreach=None,
    differentiable=False,
)

optim.Adadelta(
    params=model.parameters(),
    lr=1.0,
    rho=0.9,                   # Decay factor for running average of squared gradients
    eps=1e-6,
    weight_decay=0,
    maximize=False,
    foreach=None,
    differentiable=False,
    capturable=False,
)

optim.AdamW(model.parameters(), lr=1e-3)   # Most common choice for transformers

optim.LBFGS(
    params=model.parameters(),
    lr=1,
    max_iter=20,               # Max iterations per .step() call
    max_eval=None,             # Max function evaluations per step
    tolerance_grad=1e-7,       # Stop when gradient norm below this
    tolerance_change=1e-9,     # Stop when loss change below this
    history_size=100,          # Past gradients stored for Hessian approximation
    line_search_fn=None,       # Line search: None or 'strong_wolfe'
)

optim.Rprop(
    params=model.parameters(),
    lr=0.01,
    etas=(0.5, 1.2),           # (decrease, increase) multiplicative factors
    step_sizes=(1e-6, 50),     # (min, max) allowed step sizes
    maximize=False,
    foreach=None,
    differentiable=False,
    capturable=False,
)

# Param groups — different lr/wd per layer group (key for fine-tuning)
optimizer = optim.AdamW([
    {'params': model.stem.parameters(), 'lr': 1e-5, 'weight_decay': 0.0},
    {'params': model.layer1.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-3, 'weight_decay': 1e-2},
], lr=1e-4, weight_decay=1e-4)   # Defaults for groups without explicit values


# ==============================================================================
# SECTION 6 — LEARNING RATE SCHEDULERS
# ==============================================================================

optimizer = optim.Adam(model.parameters(), lr=0.01)

optim.lr_scheduler.StepLR(
    optimizer=optimizer,       # Optimizer to schedule
    step_size=30,              # Epochs between lr reductions
    gamma=0.1,                 # Multiplicative decay factor
    last_epoch=-1,             # Index of last epoch (-1 = auto-init)
    verbose='deprecated',      # Print lr on change
)

optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=[50, 100, 150], # Epoch indices where lr is reduced
    gamma=0.1,                 # Lr multiplied by gamma at each milestone
    last_epoch=-1,
    verbose='deprecated',
)

optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer,
    gamma=0.95,                # Multiplicative lr decay factor per epoch
    last_epoch=-1,
    verbose='deprecated',
)

optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=100,                 # Half-cycle length in epochs (or steps)
    eta_min=0,                 # Minimum lr at trough of cosine cycle
    last_epoch=-1,
    verbose='deprecated',
)

optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer=optimizer,
    T_0=10,                    # Steps for first restart cycle
    T_mult=2,                  # Multiply cycle length by T_mult after each restart
    eta_min=0,                 # Minimum lr
    last_epoch=-1,
    verbose='deprecated',
)

optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',                # 'min' for loss, 'max' for accuracy
    factor=0.1,                # Lr reduction multiplier
    patience=10,               # Epochs without improvement before reducing
    threshold=1e-4,            # Threshold for measuring improvement
    threshold_mode='rel',      # 'rel' or 'abs' threshold comparison
    cooldown=0,                # Epochs to wait after reduction before monitoring
    min_lr=0,                  # Lower bound on lr
    eps=1e-8,                  # Min lr change to apply
    verbose='deprecated',
)

optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=0.01,               # Peak lr
    total_steps=None,          # Total training steps (set or epochs×steps_per_epoch)
    epochs=None,               # Epochs (use with steps_per_epoch)
    steps_per_epoch=None,
    pct_start=0.3,             # Fraction of cycle spent increasing lr
    anneal_strategy='cos',     # 'cos' or 'linear' annealing
    cycle_momentum=True,       # Cycle momentum inversely with lr
    base_momentum=0.85,        # Lower momentum bound
    max_momentum=0.95,         # Upper momentum bound
    div_factor=25.0,           # initial_lr = max_lr / div_factor
    final_div_factor=1e4,      # final_lr = initial_lr / final_div_factor
    three_phase=False,         # Use 3 phases (up, down, annihilate)
    last_epoch=-1,
    verbose='deprecated',
)

optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda epoch: 0.95 ** epoch,  # Function returning lr multiplier
    last_epoch=-1,
    verbose='deprecated',
)

optim.lr_scheduler.LinearLR(
    optimizer=optimizer,
    start_factor=1.0 / 3,      # Start lr = base_lr * start_factor
    end_factor=1.0,            # End lr = base_lr * end_factor
    total_iters=5,             # Steps over which to linearly interpolate
    last_epoch=-1,
    verbose='deprecated',
)

optim.lr_scheduler.PolynomialLR(
    optimizer=optimizer,
    total_iters=5,             # Steps over which to decay
    power=1.0,                 # Polynomial power (1=linear, 2=quadratic)
    last_epoch=-1,
    verbose='deprecated',
)

# Chaining schedulers
from torch.optim.lr_scheduler import SequentialLR, ChainedScheduler

warmup  = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100)
cosine  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=900)
sched   = SequentialLR(
    optimizer=optimizer,
    schedulers=[warmup, cosine],# Schedulers to apply sequentially
    milestones=[100],           # Step counts where scheduler switches
    last_epoch=-1,
)


# ==============================================================================
# SECTION 7 — DATA LOADING
# ==============================================================================

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2

# ── 7.1  Dataset classes ─────────────────────────────────────────────────────

class CustomDataset(data.Dataset):
    """Base Dataset implementation. Override __len__ and __getitem__."""

    def __init__(self,
        root: str,             # Root data directory
        split: str = 'train',  # Dataset split
        transform=None,        # Transform applied to each sample
        target_transform=None, # Transform applied to each label
    ):
        self.root    = root
        self.split   = split
        self.transform = transform
        self.target_transform = target_transform
        # Load file list, labels, etc. here
        self.samples: List[Tuple] = []   # List of (path, label) pairs

    def __len__(self) -> int:
        return len(self.samples)         # Required: return dataset size

    def __getitem__(self, idx: int):
        """Required: return one sample (and label) by index."""
        path, label = self.samples[idx]
        # Load sample (image, audio, etc.)
        sample = self._load(path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
        return sample, label

    def _load(self, path): ...           # Your loading logic here


class IterableCustomDataset(data.IterableDataset):
    """For streaming datasets (TFRecord, web streams, etc.)."""

    def __init__(self, source, transform=None):
        self.source    = source
        self.transform = transform

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:          # Single worker
            for item in self.source:
                yield self.transform(item) if self.transform else item
        else:                            # Multi-worker: shard data per worker
            per_worker = len(self.source) // worker_info.num_workers
            start = worker_info.id * per_worker
            for item in self.source[start:start + per_worker]:
                yield item


# ── 7.2  DataLoader ──────────────────────────────────────────────────────────

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

loader = data.DataLoader(
    dataset=dataset,           # Dataset or IterableDataset
    batch_size=32,             # Samples per batch
    shuffle=True,              # Shuffle at every epoch (False for IterableDataset)
    sampler=None,              # Custom sampler (mutually exclusive with shuffle)
    batch_sampler=None,        # Custom batch sampler (overrides batch_size and shuffle)
    num_workers=4,             # Parallel worker processes for data loading
    collate_fn=None,           # Custom function to batch samples (None=default)
    pin_memory=True,           # Pin batches to page-locked memory (faster GPU transfer)
    drop_last=False,           # Drop final incomplete batch
    timeout=0,                 # Timeout for collecting a batch from workers
    worker_init_fn=None,       # Function called on each worker init (for seeding)
    multiprocessing_context=None,  # How to spawn workers: 'fork','spawn','forkserver'
    generator=None,            # RNG generator for shuffling
    prefetch_factor=2,         # Batches pre-loaded by each worker ahead of time
    persistent_workers=False,  # Keep workers alive between epochs (faster epoch start)
    pin_memory_device='',      # Device to pin memory to (default=CUDA default)
)

# Custom sampler examples
weighted_sampler = data.WeightedRandomSampler(
    weights=torch.ones(len(dataset)),  # Per-sample weights
    num_samples=len(dataset),          # Total samples drawn per epoch
    replacement=True,                  # Sample with replacement
    generator=None,
)
subset_sampler = data.SubsetRandomSampler(
    indices=list(range(1000)),         # Subset of indices to sample from
    generator=None,
)
sequential_sampler = data.SequentialSampler(dataset)  # No shuffling; indices in order

# Custom collate function (for variable-length sequences)
def collate_sequences(batch):
    """Pad variable-length sequences to same length."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded  = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels), lengths

# ── 7.3  Transforms (torchvision) ────────────────────────────────────────────

# v1 transforms (legacy)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.08, 1.0),              # Scale range for crop area
        ratio=(3/4, 4/3),              # Aspect ratio range
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.4,                # Max brightness change factor
        contrast=0.4,                  # Max contrast change factor
        saturation=0.4,                # Max saturation change factor
        hue=0.1,                       # Max hue change fraction
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
    transforms.ToTensor(),             # PIL/numpy [H,W,C] uint8 → float [C,H,W] in [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],    # Per-channel mean (ImageNet stats)
        std=[0.229, 0.224, 0.225],     # Per-channel std (ImageNet stats)
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# v2 transforms (recommended — works on tensors, PIL, and video)
transform_v2 = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    v2.RandomErasing(
        p=0.5,                         # Probability of applying erasing
        scale=(0.02, 0.33),            # Range of erased area fraction
        ratio=(0.3, 3.3),              # Aspect ratio range of erased area
        value=0,                       # Fill value (int, 'random', or tuple)
        inplace=False,
    ),
    v2.RandAugment(
        num_ops=2,                     # Number of augmentation operations to apply
        magnitude=9,                   # Strength of augmentations (0–30)
        num_magnitude_bins=31,
        interpolation=v2.InterpolationMode.NEAREST,
        fill=None,
    ),
    v2.ToDtype(torch.float32, scale=True),   # Normalize to [0,1] and cast
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==============================================================================
# SECTION 8 — TRAINING LOOP (COMPLETE PRODUCTION TEMPLATE)
# ==============================================================================

import time
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    """Full production training loop with AMP, gradient clipping, checkpointing."""

    def __init__(self,
        model: nn.Module,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        device: str = 'cuda',
        use_amp: bool = True,          # Automatic Mixed Precision
        grad_clip: float = 1.0,        # Gradient clipping norm
        checkpoint_dir: str = './ckpts',
    ):
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.loss_fn     = loss_fn
        self.device      = device
        self.grad_clip   = grad_clip
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # GradScaler for AMP: scales loss to prevent fp16 underflow
        self.scaler = GradScaler(
            device='cuda',             # Device for the scaler
            init_scale=2**16,          # Initial loss scale factor
            growth_factor=2.0,         # Multiply scale by this if no inf/nan for growth_interval steps
            backoff_factor=0.5,        # Multiply scale by this when inf/nan detected
            growth_interval=2000,      # Steps between scale growth attempts
            enabled=use_amp,           # Disable scaler entirely if not using AMP
        )
        self.use_amp = use_amp
        self.best_val_loss = float('inf')
        self.global_step   = 0

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device, non_blocking=True), \
                   y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(
                set_to_none=True,      # Faster than zero_(): frees memory instead of zeroing
            )

            # Automatic Mixed Precision forward pass
            with autocast(
                device_type=self.device,  # 'cuda', 'cpu', 'mps'
                dtype=torch.float16,      # Cast to this dtype inside context
                enabled=self.use_amp,
                cache_enabled=True,       # Cache dtype casts
            ):
                logits = self.model(x)
                loss   = self.loss_fn(logits, y)

            # Scaled backward + optimizer step
            self.scaler.scale(loss).backward()                  # Scale loss, then backprop
            self.scaler.unscale_(self.optimizer)                # Unscale gradients for clipping

            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=self.grad_clip,  # Maximum gradient norm
                norm_type=2.0,            # Norm type (2 = L2 norm)
                error_if_nonfinite=False, # Raise if gradients are inf/nan
            )

            self.scaler.step(self.optimizer)   # optimizer.step() if gradients are finite
            self.scaler.update()               # Update loss scale for next iteration

            total_loss    += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total         += x.size(0)
            self.global_step += 1

            if batch_idx % 100 == 0:
                print(f"  Step {batch_idx}/{len(self.train_loader)} | "
                      f"loss={loss.item():.4f} | "
                      f"scale={self.scaler.get_scale():.0f}")

        # Step schedulers that update per epoch
        if not isinstance(self.scheduler,
                          optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        return {'loss': total_loss / total,
                'acc': total_correct / total,
                'time': time.time() - t0}

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(x)
                loss   = self.loss_fn(logits, y)
            total_loss    += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total         += x.size(0)

        val_loss = total_loss / total
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        return {'loss': val_loss, 'acc': total_correct / total}

    def save_checkpoint(self, epoch: int, metrics: Dict, name: str = 'best'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),     # Model weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),   # AMP scaler state
            'metrics': metrics,
            'global_step': self.global_step,
        }, os.path.join(self.checkpoint_dir, f'{name}.pt'))

    def load_checkpoint(self, path: str, strict: bool = True):
        ckpt = torch.load(
            path,
            map_location=self.device,  # Load to target device directly
            weights_only=True,         # Only load weights (safer; avoid pickle exploit)
        )
        self.model.load_state_dict(ckpt['model_state_dict'], strict=strict)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        return ckpt['epoch'], ckpt['metrics']

    def fit(self, epochs: int):
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.validate()
            print(f"Epoch {epoch:03d} | "
                  f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} | "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"time={train_metrics['time']:.1f}s")
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, name='best')
            self.save_checkpoint(epoch, val_metrics, name='last')


# ==============================================================================
# SECTION 9 — FUNCTIONAL API (torch.nn.functional)
# ==============================================================================

x = torch.randn(8, 128)

# Activations (stateless)
F.relu(x, inplace=False)
F.leaky_relu(x, negative_slope=0.01, inplace=False)
F.prelu(x, weight=torch.tensor([0.25]))   # weight must be 1-D
F.elu(x, alpha=1.0, inplace=False)
F.selu(x, inplace=False)
F.gelu(x, approximate='none')            # 'none' or 'tanh'
F.silu(x, inplace=False)                 # Swish
F.mish(x, inplace=False)
F.sigmoid(x)
F.tanh(x)
F.hardswish(x, inplace=False)
F.hardsigmoid(x, inplace=False)
F.softmax(x, dim=1, dtype=None)
F.log_softmax(x, dim=1, dtype=None)
F.softplus(x, beta=1, threshold=20)
F.softsign(x)

# Linear
weight = torch.randn(64, 128)
bias   = torch.randn(64)
F.linear(x, weight, bias)               # y = x @ W^T + b

# Dropout
F.dropout(x, p=0.5, training=True, inplace=False)
F.dropout2d(x.unsqueeze(-1).unsqueeze(-1), p=0.5, training=True, inplace=False)
F.alpha_dropout(x, p=0.5, training=True, inplace=False)

# Normalization
F.batch_norm(
    input=x.unsqueeze(0).unsqueeze(-1),
    running_mean=torch.zeros(128),
    running_var=torch.ones(128),
    weight=None,               # Learnable gamma
    bias=None,                 # Learnable beta
    training=True,             # Use batch stats (True) or running stats (False)
    momentum=0.1,
    eps=1e-5,
)
F.layer_norm(x, normalized_shape=[128], weight=None, bias=None, eps=1e-5)
F.group_norm(x.unsqueeze(-1), num_groups=8, weight=None, bias=None, eps=1e-5)
F.instance_norm(x.unsqueeze(-1), running_mean=None, running_var=None,
                weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-5)

# Loss functions (functional)
logits = torch.randn(8, 10)
target = torch.randint(0, 10, (8,))
F.cross_entropy(logits, target, weight=None, ignore_index=-100,
                reduction='mean', label_smoothing=0.0)
F.binary_cross_entropy_with_logits(
    input=torch.randn(8), target=torch.randint(0, 2, (8,)).float(),
    weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
F.mse_loss(torch.randn(8), torch.randn(8), reduction='mean')
F.l1_loss(torch.randn(8), torch.randn(8), reduction='mean')
F.huber_loss(torch.randn(8), torch.randn(8), reduction='mean', delta=1.0)
F.nll_loss(F.log_softmax(logits, dim=1), target, weight=None,
           ignore_index=-100, reduction='mean')
F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits, dim=1),
         reduction='batchmean', log_target=False)

# Convolution
weight2d = torch.randn(32, 3, 3, 3)
F.conv2d(
    input=torch.randn(1, 3, 32, 32),
    weight=weight2d,
    bias=None,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
)

# Pooling
F.max_pool2d(torch.randn(1, 32, 16, 16), kernel_size=2, stride=2,
             padding=0, dilation=1, ceil_mode=False, return_indices=False)
F.avg_pool2d(torch.randn(1, 32, 16, 16), kernel_size=2, stride=None,
             padding=0, ceil_mode=False, count_include_pad=True)
F.adaptive_avg_pool2d(torch.randn(1, 32, 16, 16), output_size=(1, 1))

# Padding
F.pad(
    input=torch.randn(1, 3, 8, 8),
    pad=(1, 1, 1, 1),          # (left, right, top, bottom) for 2D
    mode='constant',           # 'constant','reflect','replicate','circular'
    value=0,                   # Fill value for constant mode
)

# Interpolation
F.interpolate(
    input=torch.randn(1, 64, 16, 16),
    size=None,                 # Target spatial size
    scale_factor=2,            # Upsampling multiplier
    mode='bilinear',           # 'nearest','linear','bilinear','bicubic','trilinear'
    align_corners=False,       # Align corner pixels
    recompute_scale_factor=None,
    antialias=False,
)

# Similarity
F.cosine_similarity(torch.randn(8, 128), torch.randn(8, 128), dim=1, eps=1e-8)
F.pairwise_distance(torch.randn(8, 128), torch.randn(8, 128), p=2, eps=1e-6)

# Embedding
F.embedding(
    input=torch.randint(0, 100, (8, 10)),  # Integer indices
    weight=torch.randn(100, 64),            # Embedding matrix
    padding_idx=0,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
)

# One-hot encoding
F.one_hot(
    tensor=torch.randint(0, 10, (8,)),  # Integer class indices
    num_classes=10,                      # Number of classes (-1 = auto)
)


# ==============================================================================
# SECTION 10 — ADVANCED ARCHITECTURES
# ==============================================================================

# ── 10.1  Vision Transformer (ViT) ───────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into patches and linearly project to embedding dim."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                              # (B, embed_dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)           # (B, num_patches, embed_dim)


class ViTBlock(nn.Module):
    """Transformer encoder block with pre-norm (ViT style)."""
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]  # Self-attention
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # Learnable CLS token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, embed_dim)
        x   = torch.cat([cls, x], dim=1)             # Prepend CLS token
        x   = self.pos_drop(x + self.pos_embed)      # Add positional embeddings

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])                    # CLS token → class logits


# ── 10.2  Variational Autoencoder (VAE) ──────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)   # Mean of q(z|x)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)   # Log-variance of q(z|x)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * sigma."""
        if self.training:
            std = torch.exp(0.5 * logvar)          # sigma from log-variance
            eps = torch.randn_like(std)             # N(0,1) noise
            return mu + eps * std
        return mu                                   # Deterministic at inference

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @staticmethod
    def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
        """ELBO = Reconstruction + beta * KL divergence."""
        recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl


# ── 10.3  DCGAN ──────────────────────────────────────────────────────────────

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=1, base_filters=64):
        super().__init__()
        nf = base_filters
        self.net = nn.Sequential(
            # latent → 7×7 feature map
            nn.ConvTranspose2d(latent_dim, nf*8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(nf*8), nn.ReLU(True),
            # 7×7 → 14×14
            nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*4), nn.ReLU(True),
            # 14×14 → 28×28
            nn.ConvTranspose2d(nf*4, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(lambda m: nn.init.normal_(m.weight, 0, 0.02)
                   if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) else None)

    def forward(self, z):
        return self.net(z.unsqueeze(-1).unsqueeze(-1))


class DCGANDiscriminator(nn.Module):
    def __init__(self, img_channels=1, base_filters=64):
        super().__init__()
        nf = base_filters
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)     # Raw logits for BCEWithLogitsLoss


# ── 10.4  LSTM Sequence Classifier ───────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers,
                 num_classes, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * factor, num_classes)

    def forward(self, x, lengths=None):
        emb = self.drop(self.embedding(x))

        if lengths is not None:
            # Pack padded sequences for efficient RNN processing
            packed = nn.utils.rnn.pack_padded_sequence(
                input=emb,
                lengths=lengths.cpu(),  # Lengths must be on CPU
                batch_first=True,
                enforce_sorted=False,   # Auto-sort by length if needed
            )
            out, (h_n, _) = self.lstm(packed)
            # Unpack back to padded format
            out, _ = nn.utils.rnn.pad_packed_sequence(
                sequence=out,
                batch_first=True,
                padding_value=0.0,
                total_length=None,      # Total length for padding (None=max in batch)
            )
        else:
            out, (h_n, _) = self.lstm(emb)

        # Concatenate final forward + backward hidden states
        if self.lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        return self.head(self.drop(h))


# ==============================================================================
# SECTION 11 — SAVING, LOADING & DEPLOYMENT
# ==============================================================================

model = ResNet()

# ── 11.1  State dict (recommended) ───────────────────────────────────────────

# Save
torch.save(
    obj={'model': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'epoch': 50,
         'metrics': {'val_acc': 0.92}},
    f='checkpoint.pt',             # File path or file-like object
    pickle_module=None,            # Custom pickle module
    pickle_protocol=None,          # Pickle protocol version
    _use_new_zipfile_serialization=True,  # Use zip format (smaller files)
)

# Load
checkpoint = torch.load(
    f='checkpoint.pt',
    map_location='cpu',            # Remap storage locations (e.g., GPU → CPU)
    pickle_module=None,
    weights_only=True,             # Safer: restrict to tensor/param objects only
    mmap=None,                     # Memory-map the file (for large models)
)
model.load_state_dict(
    state_dict=checkpoint['model'],
    strict=True,                   # Require exact key match (False allows partial load)
    assign=False,                  # Assign instead of copy_ (needed for meta tensors)
)

# ── 11.2  TorchScript — serialize to a static graph ─────────────────────────

# Tracing — records operations for one input
traced = torch.jit.trace(
    func=model,                    # Module or callable to trace
    example_inputs=torch.randn(1, 3, 224, 224),  # Example input(s)
    optimize=None,
    check_trace=True,              # Verify trace produces same output
    check_inputs=None,             # Additional inputs to verify against
    strict=True,                   # Strict trace mode
)
traced.save('model_traced.pt')     # Save to file

# Scripting — converts entire Python code (supports control flow)
scripted = torch.jit.script(
    obj=model,                     # Module to script (must be JIT-compatible)
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs=None,
)
scripted.save('model_scripted.pt')

# Load scripted/traced model
loaded = torch.jit.load(
    f='model_scripted.pt',
    map_location='cpu',            # Device to load onto
    _extra_files=None,             # Dict to store extra embedded files
)

# ── 11.3  ONNX export ────────────────────────────────────────────────────────

torch.onnx.export(
    model=model,                   # PyTorch model to export
    args=torch.randn(1, 3, 224, 224),  # Example input(s)
    f='model.onnx',                # Output file path
    export_params=True,            # Store trained parameters in model file
    verbose=False,                 # Print exported graph description
    training=torch.onnx.TrainingMode.EVAL,  # Export in eval mode
    input_names=['input'],         # Names for input nodes
    output_names=['output'],       # Names for output nodes
    dynamic_axes={                 # Specify dynamic (variable) dimensions
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
    opset_version=17,              # ONNX opset version to target
    do_constant_folding=True,      # Fold constant operations for optimization
    keep_initializers_as_inputs=None,
    custom_opsets=None,
)

# ── 11.4  torch.compile (PyTorch 2.0+) ───────────────────────────────────────

compiled_model = torch.compile(
    model=model,
    fullgraph=False,               # Compile entire module as one graph (strict)
    dynamic=None,                  # Dynamic shapes: None=auto, True=always, False=never
    backend='inductor',            # Compiler: 'inductor'(default),'aot_eager','eager'
    mode='default',                # Optimization: 'default','reduce-overhead','max-autotune'
    options=None,                  # Backend-specific options dict
    disable=False,                 # Disable compilation (fallback to eager)
)


# ==============================================================================
# SECTION 12 — DISTRIBUTED TRAINING
# ==============================================================================

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ── 12.1  DataParallel (single machine, multi-GPU, simple but slower) ────────

if torch.cuda.device_count() > 1:
    model_dp = nn.DataParallel(
        module=model,              # Module to parallelize
        device_ids=None,           # GPU ids to use (None=all available)
        output_device=None,        # Device for output (default=device_ids[0])
        dim=0,                     # Batch dimension to scatter/gather
    )

# ── 12.2  DistributedDataParallel (recommended for multi-GPU/multi-node) ────

def setup(rank: int, world_size: int):
    """Initialize process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',            # 'nccl' for GPU, 'gloo' for CPU
        init_method='env://',      # Init via environment variables
        world_size=world_size,     # Total number of processes
        rank=rank,                 # This process's rank
        timeout=None,              # Timeout for operations
    )
    torch.cuda.set_device(rank)    # Set default GPU for this process

def cleanup():
    dist.destroy_process_group()   # Clean up process group on exit

def train_ddp(rank: int, world_size: int, dataset):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Each process gets its own model replica
    model_ddp = ResNet().to(device)
    model_ddp = DDP(
        module=model_ddp,
        device_ids=[rank],         # GPU for this process
        output_device=rank,        # GPU to gather output on
        broadcast_buffers=True,    # Sync buffers (e.g. BN running stats)
        find_unused_parameters=False,  # Find and mark unused params (slight overhead)
        gradient_as_bucket_view=True,  # Memory optimization: use bucket memory for grads
        static_graph=False,        # Set True if graph is static (small speedup)
    )

    # Sampler ensures each worker sees different data
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,   # Number of processes
        rank=rank,                 # This process's rank
        shuffle=True,
        seed=42,
        drop_last=False,
    )
    loader = data.DataLoader(dataset, batch_size=32, sampler=sampler,
                             num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model_ddp.parameters(), lr=1e-3)

    for epoch in range(10):
        sampler.set_epoch(epoch)   # Must be called to ensure proper shuffling per epoch
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model_ddp(x), y)
            loss.backward()        # Gradients automatically averaged across processes
            optimizer.step()

        # Synchronize metric across processes
        loss_tensor = torch.tensor(loss.item(), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"Epoch {epoch}: avg_loss={loss_tensor.item():.4f}")

    cleanup()

# Launch with:
# mp.spawn(train_ddp, args=(world_size, dataset), nprocs=world_size, join=True)

# ── 12.3  Distributed communication primitives ───────────────────────────────

# (Inside a distributed context)
# dist.broadcast(tensor, src=0)            # Broadcast from rank 0 to all
# dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # Sum across all processes
# dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)  # Reduce to one process
# dist.all_gather(tensor_list, tensor)     # Gather from all to all
# dist.gather(tensor, gather_list, dst=0) # Gather to one process
# dist.scatter(tensor, scatter_list, src=0)  # Scatter from one to all
# dist.barrier()                           # Synchronization point (all must reach)


# ==============================================================================
# SECTION 13 — PERFORMANCE OPTIMIZATION
# ==============================================================================

# ── 13.1  torch.compile pipeline ─────────────────────────────────────────────

model = ResNet()

# Profile with torch.profiler to find bottlenecks
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[
        ProfilerActivity.CPU,      # Profile CPU operations
        ProfilerActivity.CUDA,     # Profile CUDA operations
    ],
    schedule=torch.profiler.schedule(
        wait=1,                    # Skip first N steps
        warmup=1,                  # Warmup steps (not recorded)
        active=3,                  # Steps to profile
        repeat=2,                  # Repeat the cycle
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    record_shapes=True,            # Record input shapes
    profile_memory=True,           # Profile memory allocation
    with_stack=True,               # Record call stack
    with_flops=True,               # Estimate FLOPs
    with_modules=True,             # Record module hierarchy
) as prof:
    for step, (x, y) in enumerate(loader):
        with record_function("model_inference"):  # Label this block in profile
            out = model(x)
        prof.step()                # Signal profiler to move to next step

print(prof.key_averages().table(
    sort_by='cuda_time_total',     # Sort column
    row_limit=10,                  # Show top N rows
))

# ── 13.2  Memory management ──────────────────────────────────────────────────

torch.cuda.memory_allocated()     # Current memory allocated to tensors (bytes)
torch.cuda.max_memory_allocated()  # Peak memory allocated since last reset
torch.cuda.memory_reserved()      # Memory reserved by caching allocator (bytes)
torch.cuda.empty_cache()          # Release cached memory back to OS (doesn't free allocated)
torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics

# Gradient checkpointing — trade compute for memory
import torch.utils.checkpoint as checkpoint

class MemoryEfficientResNet(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = checkpoint.checkpoint(
                function=block,    # Function to checkpoint
                x,                 # Arguments to function
                use_reentrant=False,  # Use non-reentrant impl (recommended)
                preserve_rng_state=True,  # Save RNG state for Dropout
            )
        return x

# ── 13.3  Channels-last memory format (faster on CUDA for CNNs) ──────────────

model = model.to(memory_format=torch.channels_last)  # NHWC layout
x = x.to(memory_format=torch.channels_last)          # Input must also be channels-last

# ── 13.4  torch.jit.script compilation inside a module ───────────────────────

# Annotate types for better scripting
@torch.jit.script
def fused_attention(q: Tensor, k: Tensor, v: Tensor, dropout_p: float = 0.0) -> Tensor:
    """Scaled dot-product attention (JIT-compiled)."""
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    return torch.matmul(weights, v)

# Flash Attention (PyTorch 2.0+ built-in)
out = F.scaled_dot_product_attention(
    query=torch.randn(8, 12, 64, 64),
    key=torch.randn(8, 12, 64, 64),
    value=torch.randn(8, 12, 64, 64),
    attn_mask=None,            # Optional boolean or float mask
    dropout_p=0.0,             # Dropout probability (training only)
    is_causal=False,           # Apply causal (autoregressive) mask
    scale=None,                # Custom scale factor (None = 1/sqrt(head_dim))
)


# ==============================================================================
# SECTION 14 — HOOKS & MODEL INSPECTION
# ==============================================================================

# ── 14.1  Forward and backward hooks ────────────────────────────────────────

activations = {}
gradients   = {}

def forward_hook(module, input, output):
    """Called after forward pass. Record output activations."""
    activations[module.__class__.__name__] = output.detach()

def backward_hook(module, grad_input, grad_output):
    """Called during backward pass. Record gradients."""
    gradients[module.__class__.__name__] = grad_output[0].detach()

# Register hooks
handle_fwd = model.layer4.register_forward_hook(forward_hook)
handle_bwd = model.layer4.register_full_backward_hook(backward_hook)

# Remove hooks when done (important to avoid memory leaks)
handle_fwd.remove()
handle_bwd.remove()

# ── 14.2  Named parameters and modules ───────────────────────────────────────

for name, param in model.named_parameters():
    print(f"{name:50s} {param.shape} requires_grad={param.requires_grad}")

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(f"Conv2d at {name}: {module.in_channels} → {module.out_channels}")

# ── 14.3  Parameter counting ─────────────────────────────────────────────────

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")


# ==============================================================================
# SECTION 15 — TRANSFER LEARNING & FINE-TUNING
# ==============================================================================

import torchvision.models as models

# ── 15.1  Load pretrained model ───────────────────────────────────────────────

backbone = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,  # Pretrained weights
)
# Other common models:
# models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
# models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False    # No gradient for frozen params

# Replace classifier head
num_classes = 5
in_features = backbone.classifier[1].in_features
backbone.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, num_classes),
)
# Only classifier params are trainable now

# Phase 2: Selective unfreezing (fine-tune last N blocks + head)
for name, param in backbone.named_parameters():
    if 'features.7' in name or 'features.8' in name or 'classifier' in name:
        param.requires_grad = True     # Unfreeze these layers

# Different lr per group for fine-tuning
optimizer = optim.AdamW([
    {'params': [p for n,p in backbone.named_parameters()
                if 'classifier' not in n and p.requires_grad], 'lr': 1e-5},
    {'params': backbone.classifier.parameters(), 'lr': 1e-3},
], weight_decay=1e-2)


# ==============================================================================
# SECTION 16 — QUANTIZATION
# ==============================================================================

import torch.quantization as quant

# ── 16.1  Post-training dynamic quantization (easiest) ───────────────────────

quantized_model = torch.quantization.quantize_dynamic(
    model=model,               # Model to quantize
    qconfig_spec={nn.Linear, nn.LSTM},  # Layers to quantize dynamically
    dtype=torch.qint8,         # Weight quantization dtype (qint8 or float16)
    mapping=None,              # Custom layer-type → quantized-type mapping
    inplace=False,             # Quantize in-place
)

# ── 16.2  Post-training static quantization ──────────────────────────────────

model.eval()
model.qconfig = quant.get_default_qconfig(
    backend='fbgemm',          # 'fbgemm' for x86 CPU, 'qnnpack' for ARM/mobile
)
quant.prepare(model, inplace=True)         # Insert observers

# Run calibration data through model (observers collect statistics)
with torch.no_grad():
    for x, _ in loader:
        model(x)
        break

quant.convert(model, inplace=True)         # Replace observers with quantized ops

# ── 16.3  Quantization-aware training (QAT) ──────────────────────────────────

model.train()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)     # Insert fake quantization ops

# Train normally with fake quantization ops active
# Then convert to quantized model for inference:
model.eval()
quant.convert(model, inplace=True)


# ==============================================================================
# SECTION 17 — MISCELLANEOUS UTILITIES
# ==============================================================================

# ── 17.1  Reproducibility ────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)                        # CPU RNG
    torch.cuda.manual_seed(seed)                   # Current GPU RNG
    torch.cuda.manual_seed_all(seed)               # All GPU RNGs
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True      # Deterministic algorithms only
    torch.backends.cudnn.benchmark = False         # Disable autotuner (non-deterministic)
    torch.use_deterministic_algorithms(
        mode=True,                                 # Raise if non-deterministic op used
        warn_only=False,                           # Warn instead of raise
    )

# ── 17.2  GPU configuration ───────────────────────────────────────────────────

torch.backends.cudnn.benchmark = True     # Autoselect fastest convolution algorithm
torch.backends.cudnn.deterministic = False

torch.cuda.memory.set_per_process_memory_fraction(
    fraction=0.9,              # Fraction of GPU memory this process can use
    device=0,                  # GPU device index
)

# Multiple GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Only expose GPUs 0 and 1

# ── 17.3  EMA (Exponential Moving Average of weights) ────────────────────────

class EMA:
    """Maintains exponential moving average of model weights for smoother inference."""

    def __init__(self,
        model: nn.Module,
        decay: float = 0.9999,     # EMA decay rate (higher = slower update, smoother)
        warmup_steps: int = 100,   # Steps before using full decay rate
    ):
        self.model  = model
        self.decay  = decay
        self.warmup = warmup_steps
        self.step   = 0
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()
                       if 'num_batches_tracked' not in k}

    def update(self):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))  # Warmup ramp
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if 'num_batches_tracked' in k: continue
                self.shadow[k] -= (1 - d) * (self.shadow[k] - v.float())

    def apply(self):
        """Apply EMA weights to model (call before inference)."""
        self.original = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=False)

    def restore(self):
        """Restore original weights (call after inference)."""
        self.model.load_state_dict(self.original, strict=False)

# ── 17.4  Gradient accumulation (simulate larger batch size) ─────────────────

accumulation_steps = 4             # Accumulate for N steps before optimizer step
optimizer.zero_grad()

for step, (x, y) in enumerate(loader):
    x, y = x.to('cuda'), y.to('cuda')
    loss = F.cross_entropy(model(x), y) / accumulation_steps  # Scale loss
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

# ── 17.5  torch.nn.utils helpers ─────────────────────────────────────────────

# Gradient clipping
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)  # Clip each grad value

# Weight norm — re-parameterize weights as (g, v) for training stability
nn.utils.weight_norm(nn.Linear(64, 64), name='weight', dim=0)
nn.utils.remove_weight_norm(module)     # Remove weight norm re-parameterization

# Spectral norm — constrain Lipschitz constant (useful for GANs)
nn.utils.spectral_norm(nn.Conv2d(64, 64, 3), name='weight', n_power_iterations=1)


# ==============================================================================
# END OF PYTORCH COMPLETE GUIDE
# Topics covered:
#   1  Tensors — creation, properties, casting, device transfer
#   2  Autograd — GradientTape equiv, custom Functions, higher-order grads
#   3  nn.Module — every layer, activation, loss, container, with all params
#   4  Custom models — ResNet, weight initialization, freezing
#   5  Optimizers — all 8 optimizers with every parameter
#   6  LR Schedulers — all schedulers, SequentialLR, OneCycleLR
#   7  DataLoader — Dataset, IterableDataset, samplers, v2 transforms
#   8  Training loop — AMP, GradScaler, gradient clipping, checkpointing
#   9  Functional API — F.* equivalent of all nn.* layers
#  10  Advanced architectures — ViT, VAE, DCGAN, LSTM with PackedSequence
#  11  Saving & deployment — state_dict, TorchScript, ONNX, torch.compile
#  12  Distributed training — DDP, DistributedSampler, communication ops
#  13  Performance — profiler, memory management, channels-last, Flash Attention
#  14  Hooks & inspection — forward/backward hooks, parameter counting
#  15  Transfer learning — pretrained models, selective unfreezing, param groups
#  16  Quantization — dynamic, static, QAT
#  17  Utilities — reproducibility, EMA, gradient accumulation, spectral norm
# ==============================================================================
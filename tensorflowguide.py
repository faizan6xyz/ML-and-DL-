# ==============================================================================
#
#   T E N S O R F L O W   —   C O M P L E T E   A D V A N C E D   G U I D E
#
#   Covers everything from tensors to distributed training to deployment
#   Every parameter, method, and pattern explained with inline comments
#
#   pip install tensorflow tensorflow-datasets tensorboard
#
# ==============================================================================

import tensorflow as tf
import numpy as np
import os, sys

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))


# ==============================================================================
# SECTION 1 — TENSORS: CREATION, TYPES, SHAPES
# ==============================================================================

# ── 1.1  Basic tensor creation ─────────────────────────────────────────────────

a = tf.constant(
    value=[[1, 2], [3, 4]],   # Python list, NumPy array, or scalar
    dtype=tf.float32,          # Data type; inferred if None
    shape=None,                # Reshape value to this shape if provided
    name='my_const',           # Optional name shown in TensorBoard graphs
)

z = tf.zeros(
    shape=(3, 4),              # Output tensor shape as list/tuple/TensorShape
    dtype=tf.float32,          # Element data type
    name=None,
)

o = tf.ones(
    shape=(2, 3),              # Shape of all-ones tensor
    dtype=tf.float32,
    name=None,
)

f = tf.fill(
    dims=(3, 3),               # Shape of the output tensor
    value=7.0,                 # Scalar value to fill every element
    name=None,
    layout=None,               # DTensor layout (advanced distributed use)
)

r = tf.random.uniform(
    shape=(4, 4),              # Output shape
    minval=0,                  # Lower bound (inclusive)
    maxval=1,                  # Upper bound (exclusive for float, inclusive for int)
    dtype=tf.float32,          # Output dtype
    seed=42,                   # Random seed for reproducibility
    name=None,
)

rn = tf.random.normal(
    shape=(4, 4),              # Output shape
    mean=0.0,                  # Mean of the normal distribution
    stddev=1.0,                # Standard deviation of the distribution
    dtype=tf.float32,
    seed=None,
    name=None,
)

trunc = tf.random.truncated_normal(
    shape=(4, 4),              # Output shape
    mean=0.0,                  # Distribution mean
    stddev=1.0,                # Standard deviation (values > 2σ are re-drawn)
    dtype=tf.float32,          # Useful for weight init: no extreme initial values
    seed=None,
    name=None,
)

eye = tf.eye(
    num_rows=4,                # Number of rows in identity matrix
    num_columns=None,          # Number of columns (square if None)
    batch_shape=None,          # Batch dimensions for batched identity matrices
    dtype=tf.float32,
    name=None,
)

rng = tf.range(
    start=0,                   # Start value (or stop if only 1 arg given)
    limit=10,                  # Stop value (exclusive)
    delta=1,                   # Step size between values
    dtype=None,                # Inferred from inputs
    name='range',
)

lin = tf.linspace(
    start=0.0,                 # Start of range (inclusive)
    stop=1.0,                  # End of range (inclusive)
    num=11,                    # Number of evenly spaced values
    name=None,
    axis=0,                    # Axis in result where values are placed
)

# ── 1.2  Tensor properties ────────────────────────────────────────────────────

t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(t.shape)                 # TensorShape — static shape when known
print(t.dtype)                 # Data type of elements
print(t.ndim)                  # Number of dimensions (rank)
print(t.device)                # Device string where tensor lives
print(t.numpy())               # Convert to NumPy array (only in eager mode)
print(tf.size(t))              # Total number of elements
print(tf.rank(t))              # Rank (number of dimensions) as tensor

# ── 1.3  Type casting ─────────────────────────────────────────────────────────

x_int  = tf.constant([1, 2, 3])
x_fp32 = tf.cast(x_int, dtype=tf.float32)  # Cast to new dtype; data is converted
x_fp16 = tf.cast(x_fp32, dtype=tf.float16) # Reduce precision for memory/speed
x_bool = tf.cast(x_int, dtype=tf.bool)     # Nonzero → True

# ── 1.4  Shape manipulation ───────────────────────────────────────────────────

t = tf.random.normal((2, 3, 4))

reshaped = tf.reshape(
    tensor=t,                  # Input tensor to reshape
    shape=(6, 4),              # New shape; one dim may be -1 to auto-compute
    name=None,
)

transposed = tf.transpose(
    a=t,                       # Input tensor
    perm=[0, 2, 1],            # Permutation of axes (None = reverse all)
    conjugate=False,           # Also conjugate complex elements
    name='transpose',
)

squeezed = tf.squeeze(
    input=tf.zeros((1, 3, 1)), # Input tensor
    axis=None,                 # Axis to remove (must be size 1); None removes all
    name=None,
)

expanded = tf.expand_dims(
    input=tf.zeros((3, 4)),    # Input tensor
    axis=0,                    # Position where new dimension is inserted
    name=None,
)

tiled = tf.tile(
    input=tf.constant([[1, 2]]),# Input tensor to tile
    multiples=[3, 2],          # Number of times to tile along each dimension
    name=None,
)

broadcast = tf.broadcast_to(
    input=tf.constant([1, 2, 3]),
    shape=(4, 3),              # Target shape; must be broadcast-compatible
    name=None,
)

# ── 1.5  Indexing and slicing ─────────────────────────────────────────────────

t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)

sliced = t[1:, ::2]            # Standard Python slicing works on tensors

gathered = tf.gather(
    params=t,                  # Tensor to gather slices from
    indices=[0, 2],            # Indices of slices to gather along axis
    validate_indices=None,
    axis=None,                 # Axis to gather along (None = axis 0)
    batch_dims=0,              # Number of batch dimensions
    name=None,
)

gathered_nd = tf.gather_nd(
    params=t,
    indices=[[0, 1], [2, 2]], # Each row is a multi-dimensional index
    batch_dims=0,
    name=None,
)

boolean_masked = tf.boolean_mask(
    tensor=t,
    mask=tf.constant([True, False, True]),  # Boolean 1-D mask
    axis=None,                 # Axis to apply mask along
    name='boolean_mask',
)

# ── 1.6  Tensor math operations ───────────────────────────────────────────────

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

tf.add(a, b)                   # Element-wise addition
tf.subtract(a, b)              # Element-wise subtraction
tf.multiply(a, b)              # Element-wise multiplication (Hadamard)
tf.divide(a, b)                # Element-wise division
tf.pow(a, 2)                   # Element-wise exponentiation
tf.sqrt(a)                     # Element-wise square root
tf.abs(a - b)                  # Element-wise absolute value
tf.exp(a)                      # Element-wise e^x
tf.math.log(a)                 # Element-wise natural log (ln)
tf.math.log(a) / tf.math.log(tf.constant(10.0))  # Log base 10

# Matrix operations
tf.matmul(a, b)                # Matrix multiplication (@ operator also works)
a @ b                          # Syntactic sugar for tf.matmul
tf.linalg.inv(a)               # Matrix inverse
tf.linalg.det(a)               # Matrix determinant
tf.linalg.eigvals(a)           # Eigenvalues
vals, vecs = tf.linalg.eigh(a) # Eigenvalues + eigenvectors (symmetric matrix)
s, u, v = tf.linalg.svd(a)    # Singular value decomposition

# Reduction operations
tf.reduce_sum(a, axis=1, keepdims=False)   # Sum along axis; keepdims keeps dim as 1
tf.reduce_mean(a, axis=0)                  # Mean along axis
tf.reduce_max(a)                           # Global max
tf.reduce_min(a, axis=1)                   # Min along axis
tf.reduce_prod(a)                          # Product of all elements
tf.math.reduce_std(a)                      # Standard deviation
tf.math.reduce_variance(a)                 # Variance

# Comparison
tf.equal(a, b)                 # Element-wise equality → bool tensor
tf.not_equal(a, b)
tf.greater(a, b)
tf.less(a, b)
tf.logical_and(a > 0, b > 5)
tf.where(
    condition=a > 2,           # Boolean condition tensor
    x=a,                       # Values where condition is True
    y=tf.zeros_like(a),        # Values where condition is False
    name=None,
)

# Sorting and argmax/argmin
tf.sort(a,
    axis=-1,                   # Axis along which to sort
    direction='ASCENDING',     # 'ASCENDING' or 'DESCENDING'
    name=None,
)
tf.argsort(a, axis=-1, direction='ASCENDING', stable=False, name=None)
tf.argmax(a, axis=1)           # Index of max value along axis
tf.argmin(a, axis=0)           # Index of min value along axis
tf.math.top_k(tf.reshape(a, [-1]), k=3)  # Top-k values and indices

# Concatenation and stacking
tf.concat([a, b], axis=0)      # Concatenate along existing axis (merges dimension)
tf.stack([a, b], axis=0)       # Stack along NEW axis (creates new dimension)
tf.unstack(a, axis=0)          # Split tensor along axis into list of tensors

# Splitting
tf.split(
    value=a,                   # Tensor to split
    num_or_size_splits=2,      # Int (equal splits) or list of sizes
    axis=0,                    # Axis to split along
    name='split',
)

# Clipping
tf.clip_by_value(
    t=a,
    clip_value_min=-1.0,       # Floor value
    clip_value_max=1.0,        # Ceiling value
    name=None,
)
tf.clip_by_norm(
    t=a,
    clip_norm=1.0,             # Maximum L2 norm
    axes=None,                 # Axes for norm computation (None=global)
    name=None,
)
tf.clip_by_global_norm(
    t_list=[a, b],             # List of tensors to clip together
    clip_norm=1.0,             # Global norm threshold
    use_norm=None,             # Precomputed global norm (None=compute it)
    name=None,
)


# ==============================================================================
# SECTION 2 — VARIABLES & AUTOMATIC DIFFERENTIATION
# ==============================================================================

# ── 2.1  tf.Variable ─────────────────────────────────────────────────────────

w = tf.Variable(
    initial_value=tf.random.normal((3, 3)), # Initial value (tensor, array, or callable)
    trainable=True,            # Include in gradient tape tracking (set False to freeze)
    validate_shape=True,       # Enforce that shape doesn't change on assign
    caching_device=None,       # Device for caching variable reads
    name='weights',            # Variable name (shown in logs and TensorBoard)
    variable_def=None,         # For deserialization only
    dtype=None,                # Inferred from initial_value if None
    import_scope=None,         # Deprecated
    constraint=None,           # Callable applied after gradient update (e.g. non_neg)
    synchronization=tf.VariableSynchronization.AUTO,  # Sync policy in distributed training
    aggregation=tf.VariableAggregation.NONE,           # How to aggregate variable in dist. training
    shape=None,                # Override shape if needed
)

# Variable operations
w.assign(tf.ones((3, 3)))                  # Replace value in-place
w.assign_add(tf.ones((3, 3)) * 0.01)      # Add to variable in-place (no new allocation)
w.assign_sub(tf.ones((3, 3)) * 0.01)      # Subtract from variable in-place
w_val = w.numpy()                          # Read value as NumPy
w.read_value()                             # Read as tensor

# ── 2.2  GradientTape — automatic differentiation ────────────────────────────

x = tf.Variable(3.0)
with tf.GradientTape(
    persistent=False,          # If True, tape can be queried multiple times (more memory)
    watch_all_variables=True,  # Automatically watch all trainable variables
) as tape:
    tape.watch(x)              # Explicitly watch non-Variable tensors
    y = x ** 2 + 3 * x + 1

dy_dx = tape.gradient(
    target=y,                  # Scalar or tensor to differentiate
    sources=x,                 # Variable(s) or tensor(s) to differentiate w.r.t.
    output_gradients=None,     # Upstream gradients for vector-Jacobian product
    unconnected_gradients=tf.UnconnectedGradients.NONE,  # Value when graph is disconnected
)

# Higher-order gradients (second derivative)
x = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3
    dy_dx  = tape1.gradient(y, x)   # First derivative
d2y_dx2 = tape2.gradient(dy_dx, x)  # Second derivative (Hessian diagonal element)

# Gradient of multiple outputs w.r.t. multiple inputs (Jacobian)
x = tf.Variable([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    y = x * x
jac = tape.jacobian(
    target=y,                  # Output tensor (may be non-scalar)
    sources=x,                 # Input variable
    unconnected_gradients=tf.UnconnectedGradients.NONE,
    parallel_iterations=None,  # Parallel jacobian computation (None=auto)
    experimental_use_pfor=True,# Use pfor for vectorized Jacobian (faster)
)

# Custom gradient
@tf.custom_gradient
def log1pexp(x):
    """Numerically stable log(1 + exp(x)) with custom grad."""
    e = tf.exp(x)
    def grad(upstream):
        return upstream * (1.0 - 1.0 / (1.0 + e))  # Custom backward pass
    return tf.math.log(1.0 + e), grad


# ==============================================================================
# SECTION 3 — tf.data PIPELINE (EFFICIENT INPUT)
# ==============================================================================

# ── 3.1  Dataset creation ────────────────────────────────────────────────────

# From tensors (loads everything into memory)
ds = tf.data.Dataset.from_tensors(
    tensors=(tf.constant([1, 2, 3]),),  # Single element; entire tensor = one item
)

ds = tf.data.Dataset.from_tensor_slices(
    tensors={'x': np.random.randn(100, 10).astype(np.float32),
             'y': np.random.randint(0, 2, 100).astype(np.int32)},
    # Each row of array becomes one dataset element; supports dict, tuple, array
)

# From Python generator (for large/dynamic data)
def data_gen():
    for i in range(100):
        yield np.random.randn(10).astype(np.float32), np.int32(i % 2)

ds = tf.data.Dataset.from_generator(
    generator=data_gen,        # Python generator or callable returning one
    output_signature=(         # Define types/shapes of each yielded element
        tf.TensorSpec(shape=(10,), dtype=tf.float32),
        tf.TensorSpec(shape=(),   dtype=tf.int32),
    ),
    args=None,                 # Arguments passed to generator
)

# From TFRecord files (recommended for large datasets)
raw_ds = tf.data.TFRecordDataset(
    filenames=['data.tfrecord'],     # File or list of files to read
    compression_type='',             # '' none, 'ZLIB', 'GZIP'
    buffer_size=None,                # Read buffer in bytes (None=auto)
    num_parallel_reads=None,         # Parallel file reading threads
    name=None,
)

# Range dataset
ds_range = tf.data.Dataset.range(
    *args,                     # Same as Python range(start, stop, step)
    output_type=tf.int64,      # Output dtype
    name=None,
)

# Zip multiple datasets together
ds_a = tf.data.Dataset.from_tensor_slices(np.arange(10))
ds_b = tf.data.Dataset.from_tensor_slices(np.arange(10) * 2)
ds_zipped = tf.data.Dataset.zip(
    datasets=(ds_a, ds_b),     # Tuple/dict of datasets to combine element-wise
    name=None,
)

# ── 3.2  Dataset transformations ─────────────────────────────────────────────

ds = tf.data.Dataset.from_tensor_slices(
    np.random.randn(1000, 10).astype(np.float32))

# map — apply function to each element
ds = ds.map(
    map_func=lambda x: x * 2,         # Function to apply to each element
    num_parallel_calls=tf.data.AUTOTUNE,# Parallel threads (AUTOTUNE=system decides)
    deterministic=True,                # Preserve element ordering when parallel
    name=None,
)

# filter — keep only elements matching condition
ds = ds.filter(
    predicate=lambda x: tf.reduce_mean(x) > 0,  # Boolean-returning function
    name=None,
)

# batch — group consecutive elements
ds = ds.batch(
    batch_size=32,             # Number of elements per batch
    drop_remainder=False,      # Drop final batch if smaller than batch_size
    num_parallel_calls=None,   # Parallel batching (rarely needed)
    deterministic=None,
    name=None,
)

# shuffle — randomize element order
ds = ds.shuffle(
    buffer_size=1000,          # Elements loaded into shuffle buffer; larger=better mixing
    seed=42,                   # Random seed for reproducibility
    reshuffle_each_iteration=True,  # Re-shuffle every epoch
    name=None,
)

# repeat — cycle through dataset
ds = ds.repeat(
    count=None,                # Times to repeat (None=infinite)
    name=None,
)

# prefetch — overlap data loading with model training
ds = ds.prefetch(
    buffer_size=tf.data.AUTOTUNE, # Elements to prefetch (AUTOTUNE=dynamic)
    name=None,
)

# flat_map — map then flatten one level
ds = ds.flat_map(
    map_func=lambda x: tf.data.Dataset.from_tensors(x),
    num_parallel_calls=None,
    deterministic=None,
    name=None,
)

# interleave — parallel reading from multiple sources (key for TFRecord)
ds = tf.data.Dataset.from_tensor_slices(['file1', 'file2']).interleave(
    map_func=lambda f: tf.data.TextLineDataset(f),  # Function returning dataset
    cycle_length=2,            # Number of input elements to process concurrently
    block_length=1,            # Consecutive elements per input before cycling
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
    name=None,
)

# take / skip / shard
ds.take(100)                   # Take first N elements
ds.skip(50)                    # Skip first N elements
ds.shard(
    num_shards=4,              # Total number of shards
    index=0,                   # This worker's shard index
    name=None,
)

# window — sliding/tumbling window over sequences
ds.window(
    size=5,                    # Window size (number of elements per window)
    shift=1,                   # Step between windows
    stride=1,                  # Stride between window elements
    drop_remainder=True,       # Drop windows smaller than size
    name=None,
)

# cache — cache dataset in memory or on disk
ds.cache(
    filename='',               # Path for disk cache ('' = in-memory cache)
    name=None,
)

# ── 3.3  Complete production input pipeline ───────────────────────────────────

def build_pipeline(file_pattern, batch_size, training=True):
    """Production-ready tf.data pipeline with all best practices."""

    feature_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_fn(example_proto):
        """Parse a single TFRecord example."""
        parsed = tf.io.parse_single_example(
            serialized=example_proto,  # Serialized TFRecord proto string
            features=feature_desc,     # Feature description dict
            example_names=None,
            name=None,
        )
        image = tf.io.decode_jpeg(
            contents=parsed['image'],  # JPEG-encoded byte string
            channels=3,                # Output channels (0=auto, 1=gray, 3=RGB)
            ratio=1,                   # Downscaling ratio during decode
            fancy_upscaling=True,
            try_recover_truncated=False,
            acceptable_fraction=1,
            dct_method='',
            name=None,
        )
        image = tf.image.resize(
            images=image,
            size=(224, 224),           # Target [height, width]
            method='bilinear',         # Interpolation: 'bilinear','nearest','bicubic','lanczos3'
            preserve_aspect_ratio=False,
            antialias=False,
            name=None,
        )
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        return image, parsed['label']

    def augment(image, label):
        """Training augmentations."""
        image = tf.image.random_flip_left_right(image, seed=None)
        image = tf.image.random_brightness(image, max_delta=0.2, seed=None)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2, seed=None)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2, seed=None)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    files = tf.data.Dataset.list_files(
        file_pattern=file_pattern,  # Glob pattern for files
        shuffle=training,           # Shuffle file order (important for training)
        seed=None,
        name=None,
    )
    ds = files.interleave(
        tf.data.TFRecordDataset, cycle_length=8,
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ==============================================================================
# SECTION 4 — BUILDING MODELS
# ==============================================================================

from tensorflow import keras
from tensorflow.keras import layers, Model, Input

# ── 4.1  Sequential API — simple linear stack ────────────────────────────────

model = keras.Sequential(
    layers=[                   # Optional list of layers to add immediately
        layers.Input(shape=(784,)),            # Defines input shape for the model
        layers.Dense(512, activation='relu'),  # Fully connected layer
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax'),
    ],
    name='mlp_model',          # Name used in logs and saved model
)

# Add layers after construction
model.add(layers.Dense(10))    # Append a layer to the existing sequential model

# ── 4.2  Functional API — arbitrary DAG architectures ────────────────────────

def build_functional_model(input_dim=784, num_classes=10):
    inputs = Input(
        shape=(input_dim,),    # Input shape (excluding batch); required for Functional API
        batch_size=None,       # Fixed batch size (None=variable)
        dtype=None,
        sparse=False,          # Whether input is a SparseTensor
        ragged=False,          # Whether input is a RaggedTensor
        name='input',
    )
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Skip connection (residual)
    residual = layers.Dense(256)(x)                # Project to match dimensions
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Add()([x, residual])                # Add residual
    x = layers.LayerNormalization()(x)

    # Multi-head output
    class_out = layers.Dense(num_classes, activation='softmax', name='class')(x)
    conf_out  = layers.Dense(1, activation='sigmoid', name='confidence')(x)

    return Model(
        inputs=inputs,
        outputs=[class_out, conf_out],  # Can be dict, list, or single tensor
        name='functional_model',
    )

# ── 4.3  Subclassing API — maximum flexibility ───────────────────────────────

class ResidualBlock(layers.Layer):
    """Custom layer with learnable skip connection."""

    def __init__(self,
        units,                 # Number of hidden units
        dropout_rate=0.1,      # Dropout probability
        **kwargs               # Pass name, dtype, etc. to parent Layer
    ):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        """Create weights lazily on first call (input shape is known here)."""
        self.dense1 = layers.Dense(self.units, activation='relu')
        self.dense2 = layers.Dense(self.units)
        self.proj   = layers.Dense(self.units)  # Project input if shape mismatch
        self.bn     = layers.BatchNormalization()
        self.drop   = layers.Dropout(self.dropout_rate)
        super().build(input_shape)              # Mark layer as built

    def call(self,
        inputs,                # Input tensor
        training=False,        # Training mode flag (affects BN and Dropout)
    ):
        residual = self.proj(inputs)
        x = self.dense1(inputs)
        x = self.bn(x, training=training)
        x = self.drop(x, training=training)
        x = self.dense2(x)
        return tf.nn.relu(x + residual)         # Residual connection with ReLU

    def get_config(self):
        """Required for serialization (model saving/loading)."""
        config = super().get_config()
        config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return config


class AdvancedModel(Model):
    """Full model subclass with custom training logic."""

    def __init__(self,
        num_classes=10,
        hidden_units=[512, 256, 128],
        dropout_rate=0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.blocks = [ResidualBlock(u, dropout_rate) for u in hidden_units]
        self.head   = layers.Dense(num_classes, activation='softmax')
        self.flatten = layers.Flatten()

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        for block in self.blocks:
            x = block(x, training=training)     # Pass training flag through
        return self.head(x)

    def get_config(self):
        return {'num_classes': self.head.units}


# ── 4.4  Custom Training Loop ────────────────────────────────────────────────

class Trainer:
    """Production-grade custom training loop with metrics and callbacks."""

    def __init__(self, model, learning_rate=1e-3):
        self.model     = model
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn   = keras.losses.SparseCategoricalCrossentropy()
        # Metric objects maintain running averages automatically
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_acc  = keras.metrics.SparseCategoricalAccuracy(name='train_acc')
        self.val_loss   = keras.metrics.Mean(name='val_loss')
        self.val_acc    = keras.metrics.SparseCategoricalAccuracy(name='val_acc')

    @tf.function(
        input_signature=None,  # Concrete input specs; None = trace on first call
        reduce_retracing=False,# Avoid retracing for slightly different inputs
        jit_compile=False,     # Enable XLA compilation for this function
    )
    def train_step(self, x, y):
        """Single training step — compiled to graph for speed."""
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
            # Add regularization losses from layers (e.g. kernel_regularizer)
            loss += tf.add_n(self.model.losses) if self.model.losses else 0.0

        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Gradient clipping to prevent exploding gradients
        gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=1.0)

        self.optimizer.apply_gradients(
            grads_and_vars=zip(gradients, self.model.trainable_variables),
        )
        self.train_loss.update_state(loss)
        self.train_acc.update_state(y, predictions)
        return loss

    @tf.function
    def val_step(self, x, y):
        """Single validation step — no gradient computation."""
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)
        self.val_loss.update_state(loss)
        self.val_acc.update_state(y, predictions)

    def fit(self, train_ds, val_ds, epochs=10):
        for epoch in range(epochs):
            # Reset metrics at start of each epoch
            for m in [self.train_loss, self.train_acc,
                      self.val_loss, self.val_acc]:
                m.reset_state()

            for x_batch, y_batch in train_ds:
                self.train_step(x_batch, y_batch)

            for x_batch, y_batch in val_ds:
                self.val_step(x_batch, y_batch)

            print(f"Epoch {epoch+1:03d} | "
                  f"loss={self.train_loss.result():.4f} "
                  f"acc={self.train_acc.result():.4f} | "
                  f"val_loss={self.val_loss.result():.4f} "
                  f"val_acc={self.val_acc.result():.4f}")


# ==============================================================================
# SECTION 5 — KERAS LAYERS: EVERY IMPORTANT LAYER WITH ALL PARAMETERS
# ==============================================================================

# ── 5.1  Core layers ─────────────────────────────────────────────────────────

dense = layers.Dense(
    units=256,
    activation='relu',          # String, callable, or None
    use_bias=True,
    kernel_initializer='glorot_uniform', # He_normal for relu, glorot for tanh/sigmoid
    bias_initializer='zeros',
    kernel_regularizer=keras.regularizers.L2(1e-4),   # L2 on weights
    bias_regularizer=None,
    activity_regularizer=keras.regularizers.L1(1e-5), # L1 on output activations
    kernel_constraint=keras.constraints.MaxNorm(3.0), # Clip weight norm
    bias_constraint=None,
    lora_rank=None,              # Low-rank adaptation rank (efficient fine-tuning)
)

# ── 5.2  Convolutional layers ────────────────────────────────────────────────

conv2d = layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',             # 'valid'=no padding, 'same'=zero-pad to same output size
    data_format='channels_last',# NHWC ('channels_last') or NCHW ('channels_first')
    dilation_rate=(1, 1),       # Atrous/dilated conv: gaps between kernel elements
    groups=1,                   # Grouped conv; in_channels/groups filters per group
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)

conv_transpose = layers.Conv2DTranspose(
    filters=32,
    kernel_size=4,
    strides=2,                  # Upsampling factor (strides > 1 increase spatial size)
    padding='same',
    output_padding=None,        # Extra output pixels added to one side
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)

# Depthwise separable — fewer params, efficient
sep_conv = layers.SeparableConv2D(
    filters=128,
    kernel_size=3,
    strides=1,
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    depth_multiplier=1,         # Output channels per input channel in depthwise step
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform', # Depthwise filter initializer
    pointwise_initializer='glorot_uniform', # Pointwise (1x1) filter initializer
    bias_initializer='zeros',
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
)

# ── 5.3  Pooling layers ──────────────────────────────────────────────────────

layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None)
layers.AveragePooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None)
layers.GlobalAveragePooling2D(data_format=None, keepdims=False)  # Spatial → vector
layers.GlobalMaxPooling2D(data_format=None, keepdims=False)

# ── 5.4  Normalization layers ────────────────────────────────────────────────

bn = layers.BatchNormalization(
    axis=-1,                   # Channel axis; -1 for channels_last, 1 for channels_first
    momentum=0.99,             # Exponential moving average momentum (higher=slower update)
    epsilon=1e-3,              # Numerical stability constant added to variance
    center=True,               # Learn beta (additive shift)
    scale=True,                # Learn gamma (multiplicative scale)
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    synchronized=False,        # Sync stats across all devices (multi-GPU/TPU)
)

ln = layers.LayerNormalization(
    axis=-1,                   # Axis(es) to normalize (usually last)
    epsilon=1e-3,
    center=True,               # Learn additive bias
    scale=True,                # Learn multiplicative scale
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
)

gn = layers.GroupNormalization(
    groups=8,                  # Number of groups to divide channels into
    axis=-1,                   # Channel axis
    epsilon=1e-3,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
)

# ── 5.5  Recurrent layers ────────────────────────────────────────────────────

lstm = layers.LSTM(
    units=256,
    activation='tanh',
    recurrent_activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',  # Orthogonal init stabilizes RNN training
    bias_initializer='zeros',
    unit_forget_bias=True,     # Forget gate bias = 1 initially (prevents forgetting early on)
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,               # Fraction of units to drop for input connections
    recurrent_dropout=0.0,     # Fraction to drop for recurrent state connections
    return_sequences=True,     # True = output at every step; False = last step only
    return_state=False,        # Also return (hidden_state, cell_state)
    go_backwards=False,        # Process sequence in reverse (for BiLSTM)
    stateful=False,            # Use end-state of batch i as start-state of batch i+1
    unroll=False,              # Unroll loop (faster for fixed-length short sequences)
)

# Bidirectional LSTM wrapper
bilstm = layers.Bidirectional(
    layer=layers.LSTM(128, return_sequences=True), # Recurrent layer to wrap
    merge_mode='concat',       # How to combine forward+backward: 'concat','sum','mul','ave',None
    weights=None,              # Initial weights for the bidirectional wrapper
)

# ── 5.6  Attention layers ────────────────────────────────────────────────────

mha = layers.MultiHeadAttention(
    num_heads=8,               # Number of attention heads
    key_dim=64,                # Key/query dimension per head; total = num_heads * key_dim
    value_dim=None,            # Value dimension per head (defaults to key_dim)
    dropout=0.1,               # Dropout on attention weights
    use_bias=True,
    output_shape=None,         # Reshape output projection
    attention_axes=None,       # Axes to attend over (None = all sequence axes)
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    seed=None,
)

# Usage: query, key, value can all be the same for self-attention
# output = mha(query=x, value=x, key=x, attention_mask=None,
#              return_attention_scores=False, training=False,
#              use_causal_mask=False)

# ── 5.7  Embedding layer ─────────────────────────────────────────────────────

emb = layers.Embedding(
    input_dim=10000,           # Vocabulary size
    output_dim=256,            # Embedding dimension
    embeddings_initializer='uniform',
    embeddings_regularizer=None,
    embeddings_constraint=None,
    mask_zero=True,            # Mask index 0 as padding (propagates mask downstream)
    weights=None,              # Optional pre-trained embedding matrix
    lora_rank=None,
)

# ── 5.8  Reshaping / utility layers ─────────────────────────────────────────

layers.Flatten(data_format=None)
layers.Reshape(target_shape=(4, 32))      # Exclude batch dim from target shape
layers.Permute(dims=(2, 1))               # Permute axes (1-indexed, excludes batch)
layers.RepeatVector(n=5)                  # Repeat input n times along new axis
layers.Lambda(lambda x: x * 2,           # Apply arbitrary expression
    output_shape=None,                    # Output shape if not inferrable
    mask=None,
    arguments=None,
)

# Merge layers
layers.Add()                              # Element-wise sum of a list of inputs
layers.Subtract()                         # Element-wise difference of 2 inputs
layers.Multiply()                         # Element-wise product
layers.Average()                          # Element-wise average
layers.Maximum()                          # Element-wise maximum
layers.Minimum()                          # Element-wise minimum
layers.Concatenate(axis=-1)               # Concatenate along given axis
layers.Dot(axes=1, normalize=False)       # Dot product between two inputs

# Dropout variants
layers.Dropout(rate=0.5, noise_shape=None, seed=None)
layers.SpatialDropout2D(rate=0.5, data_format=None)  # Drop entire feature maps (channels)
layers.GaussianDropout(rate=0.5)          # Multiply by Gaussian noise instead of zeroing
layers.GaussianNoise(stddev=0.1)          # Add Gaussian noise during training (augmentation)
layers.ActivityRegularization(l1=0.0, l2=0.0)  # Apply regularization on layer output


# ==============================================================================
# SECTION 6 — LOSS FUNCTIONS (COMPLETE)
# ==============================================================================

from tensorflow.keras import losses

# Regression
losses.MeanSquaredError(reduction='sum_over_batch_size', name='mse', dtype=None)
losses.MeanAbsoluteError(reduction='sum_over_batch_size', name='mae', dtype=None)
losses.MeanAbsolutePercentageError()      # MAPE: 100 * |y-ŷ|/|y|
losses.MeanSquaredLogarithmicError()      # MSLE: useful for targets with exponential scale
losses.Huber(delta=1.0, reduction='sum_over_batch_size', name='huber', dtype=None)
losses.LogCosh(reduction='sum_over_batch_size', name='logcosh', dtype=None)

# Classification
losses.BinaryCrossentropy(
    from_logits=False,         # If True, apply sigmoid internally (more numerically stable)
    label_smoothing=0.0,       # Smooth labels toward 0.5 (regularizes overconfidence)
    axis=-1,
    reduction='sum_over_batch_size',
    name='bce',
    dtype=None,
)

losses.CategoricalCrossentropy(
    from_logits=False,         # If True, apply softmax internally
    label_smoothing=0.0,       # Label smoothing epsilon
    axis=-1,                   # Class probability axis
    reduction='sum_over_batch_size',
    name='cce',
    dtype=None,
)

losses.SparseCategoricalCrossentropy(
    from_logits=False,         # Accepts raw logits when True (skip manual softmax)
    ignore_class=None,         # Class to ignore (e.g. padding class in segmentation)
    reduction='sum_over_batch_size',
    name='scce',
    dtype=None,
)

losses.KLDivergence(reduction='sum_over_batch_size', name='kld', dtype=None)
losses.Hinge(reduction='sum_over_batch_size', name='hinge', dtype=None)
losses.SquaredHinge(reduction='sum_over_batch_size', name='sq_hinge', dtype=None)
losses.CategoricalHinge(reduction='sum_over_batch_size', name='cat_hinge', dtype=None)

# Custom loss function
def focal_loss(alpha=0.25, gamma=2.0):
    """Focal loss for class-imbalanced detection tasks."""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce   = -y_true * tf.math.log(y_pred) - (1-y_true) * tf.math.log(1-y_pred)
        p_t   = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)
    return loss_fn


# ==============================================================================
# SECTION 7 — OPTIMIZERS (COMPLETE)
# ==============================================================================

from tensorflow.keras import optimizers

# All optimizers share these base parameters:
# - learning_rate: step size (float or LRSchedule)
# - weight_decay: decoupled L2 regularization
# - clipnorm: clip gradient by norm per tensor
# - global_clipnorm: clip combined gradient norm across all tensors
# - clipvalue: clip gradient values elementwise
# - use_ema: exponential moving average of weights
# - ema_momentum: EMA decay factor

opt = optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,              # Velocity accumulation factor (0=pure GD)
    nesterov=True,             # Look-ahead gradient step before update
    weight_decay=1e-4,
    clipnorm=None,
    global_clipnorm=1.0,       # Clip global gradient norm; prevents exploding gradients
)

opt = optimizers.Adam(
    learning_rate=1e-3,
    beta_1=0.9,                # Decay for first moment estimate (momentum)
    beta_2=0.999,              # Decay for second moment estimate (squared gradient)
    epsilon=1e-7,              # Numerical stability in denominator
    amsgrad=False,             # AMSGrad variant: uses max of past squared gradients
    weight_decay=None,
    clipnorm=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
)

opt = optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=0.004,        # Decoupled weight decay (not mixed into gradient)
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
    amsgrad=False,
)

opt = optimizers.RMSprop(
    learning_rate=1e-3,
    rho=0.9,                   # Decay factor for running average of squared gradients
    momentum=0.0,              # Momentum for RMSprop updates
    epsilon=1e-7,
    centered=False,            # Divide by centered RMS (variance-normalized) if True
)

opt = optimizers.Adagrad(
    learning_rate=0.01,
    initial_accumulator_value=0.1, # Starting value for gradient accumulator
    epsilon=1e-7,
)

opt = optimizers.Adadelta(
    learning_rate=1.0,
    rho=0.95,                  # Decay factor for running gradient and update
    epsilon=1e-7,
)

opt = optimizers.Nadam(        # Adam with Nesterov momentum
    learning_rate=2e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
)

opt = optimizers.Ftrl(         # Follow-The-Regularized-Leader; good for sparse features
    learning_rate=0.001,
    learning_rate_power=-0.5,  # Controls decay of learning rate
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    l2_shrinkage_regularization_strength=0.0,
    beta=0.0,
)

# Learning rate schedules
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,  # Starting learning rate
    decay_steps=10000,           # Steps until rate is multiplied by decay_rate
    decay_rate=0.96,             # Multiplicative decay factor
    staircase=False,             # True = discrete steps; False = continuous decay
    name=None,
)

lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,  # Starting learning rate
    decay_steps=1000,            # Steps to decay over
    alpha=0.0,                   # Minimum lr = alpha * initial_learning_rate
    name=None,
    warmup_target=None,          # Warmup to this value before cosine decay
    warmup_steps=0,              # Number of warmup steps
)

lr_schedule = optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=1000,      # Steps in first restart cycle
    t_mul=2.0,                   # Multiply steps per cycle by this factor
    m_mul=1.0,                   # Multiply initial lr per cycle by this factor
    alpha=0.0,                   # Minimum lr fraction
    name=None,
)

lr_schedule = optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    end_learning_rate=1e-5,      # Final learning rate value
    power=1.0,                   # Polynomial power (1=linear, 2=quadratic)
    cycle=False,                 # Cycle after decay_steps
    name=None,
)

# Piecewise constant (manual lr schedule)
lr_schedule = optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[3000, 7000],     # Step boundaries where lr changes
    values=[1e-2, 1e-3, 1e-4],  # lr values: len(values) = len(boundaries) + 1
    name=None,
)


# ==============================================================================
# SECTION 8 — METRICS
# ==============================================================================

from tensorflow.keras import metrics

# Regression metrics
metrics.MeanSquaredError(name='mse', dtype=None)
metrics.RootMeanSquaredError(name='rmse', dtype=None)
metrics.MeanAbsoluteError(name='mae', dtype=None)
metrics.MeanAbsolutePercentageError(name='mape', dtype=None)
metrics.CosineSimilarity(axis=-1, name='cosine', dtype=None)

# Classification metrics
metrics.Accuracy(name='accuracy', dtype=None)
metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5, dtype=None)
metrics.CategoricalAccuracy(name='cat_accuracy', dtype=None)
metrics.SparseCategoricalAccuracy(name='sparse_cat_accuracy', dtype=None)

metrics.Precision(
    thresholds=None,           # Float or list; None = 0.5
    top_k=None,                # Use top-k predictions
    class_id=None,             # Restrict to specific class
    name='precision',
    dtype=None,
)

metrics.Recall(
    thresholds=None,
    top_k=None,
    class_id=None,
    name='recall',
    dtype=None,
)

metrics.AUC(
    num_thresholds=200,        # Number of thresholds to evaluate ROC/PR curve
    curve='ROC',               # 'ROC' (TPR vs FPR) or 'PR' (Precision vs Recall)
    summation_method='interpolation', # 'interpolation','minoring','majoring'
    thresholds=None,
    multi_label=False,         # Multi-label classification
    num_labels=None,
    label_weights=None,
    from_logits=False,
    name='auc',
    dtype=None,
)

metrics.F1Score(
    average='macro',           # 'micro','macro','weighted'; how to aggregate classes
    threshold=None,
    name='f1',
    dtype=None,
)

# Confusion matrix based
metrics.TruePositives(thresholds=None, name='tp', dtype=None)
metrics.TrueNegatives(thresholds=None, name='tn', dtype=None)
metrics.FalsePositives(thresholds=None, name='fp', dtype=None)
metrics.FalseNegatives(thresholds=None, name='fn', dtype=None)

# Custom metric
class TopKAccuracy(metrics.Metric):
    def __init__(self, k=5, name='top_k_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.total   = self.add_weight('total', initializer='zeros')  # Sum of correct
        self.count   = self.add_weight('count', initializer='zeros')  # Total samples

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, self.k)
        self.total.assign_add(tf.reduce_sum(correct))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total / self.count     # Average top-k accuracy

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


# ==============================================================================
# SECTION 9 — CALLBACKS (COMPLETE)
# ==============================================================================

from tensorflow.keras import callbacks

cb_checkpoint = callbacks.ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}_{val_loss:.4f}.keras',
    monitor='val_loss',        # Metric to monitor for saving decision
    verbose=1,                 # Print message when saving
    save_best_only=True,       # Save only when monitored metric improves
    save_weights_only=False,   # Save full model (False) or just weights (True)
    mode='min',                # 'min' for loss, 'max' for accuracy
    save_freq='epoch',         # 'epoch' or integer (every N batches)
    initial_value_threshold=None, # Only save if metric better than this baseline
    options=None,              # tf.train.CheckpointOptions
)

cb_early = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,            # Minimum change qualifying as improvement
    patience=15,               # Epochs without improvement before stopping
    verbose=1,
    mode='min',
    baseline=None,             # Must beat this to count as improvement
    restore_best_weights=True, # Restore weights from best epoch after stopping
    start_from_epoch=10,       # Ignore early stopping for first N epochs
)

cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,                # Multiply lr by this factor when triggered
    patience=5,
    verbose=1,
    mode='min',
    min_delta=1e-4,
    cooldown=2,                # Epochs to wait after reduction before monitoring resumes
    min_lr=1e-7,               # Minimum lr floor
)

cb_tb = callbacks.TensorBoard(
    log_dir='./tf_logs',       # Directory for TensorBoard event files
    histogram_freq=1,          # Epoch frequency to log weight histograms
    write_graph=True,          # Write computation graph
    write_images=True,         # Write model weight images
    write_steps_per_second=True, # Log throughput metric
    update_freq='epoch',       # When to write: 'batch', 'epoch', or integer
    profile_batch=(2, 5),      # Profile batches 2–5 for performance analysis
    embeddings_freq=0,         # Epoch frequency to visualize embeddings
    embeddings_metadata=None,  # Path to metadata file for embedding visualization
)

cb_csv = callbacks.CSVLogger(
    filename='training_log.csv',  # File to append training metrics each epoch
    separator=',',             # Column separator character
    append=False,              # If True, append to existing file
)

cb_terminate = callbacks.TerminateOnNaN()  # Stop training if loss becomes NaN

cb_lr_scheduler = callbacks.LearningRateScheduler(
    schedule=lambda epoch, lr: lr * 0.95 ** epoch,  # Function(epoch, lr) → new_lr
    verbose=1,                 # Print updated lr each epoch
)

cb_nan = callbacks.TerminateOnNaN()       # Immediately stop if NaN loss detected

# Custom callback
class GradientMonitor(callbacks.Callback):
    """Log gradient norms to TensorBoard for debugging."""

    def __init__(self, val_data, log_dir='./logs/gradients'):
        super().__init__()
        self.val_data = val_data
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        x, y = next(iter(self.val_data))
        with tf.GradientTape() as tape:
            loss = self.model.compiled_loss(
                y, self.model(x, training=False))
        grads = tape.gradient(loss, self.model.trainable_variables)
        with self.writer.as_default():
            for grad, var in zip(grads, self.model.trainable_variables):
                if grad is not None:
                    tf.summary.scalar(
                        f'grad_norm/{var.name}',
                        tf.linalg.norm(grad),
                        step=epoch,
                    )


# ==============================================================================
# SECTION 10 — MODEL COMPILATION & TRAINING
# ==============================================================================

model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10),
])

model.compile(
    optimizer=optimizers.Adam(1e-3),  # Optimizer instance or string
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        metrics.SparseCategoricalAccuracy(name='acc'),
        metrics.SparseTopKCategoricalAccuracy(k=3, name='top3'),
    ],
    loss_weights=None,         # Per-output loss weights for multi-output models
    weighted_metrics=None,     # Metrics computed with sample_weight
    run_eagerly=False,         # Eager mode (slower but debuggable)
    steps_per_execution=1,     # Steps to fuse per tf.function call (TPU optimization)
    jit_compile='auto',        # XLA JIT compilation: True, False, 'auto'
    auto_scale_loss=True,      # Auto loss scaling for mixed-precision training
)

# Model summary
model.summary(
    line_length=100,           # Characters per line
    positions=None,            # Column widths as fractions
    print_fn=print,            # Custom print function
    expand_nested=True,        # Show weights of nested models
    show_trainable=True,       # Show trainable/non-trainable column
    layer_range=None,          # Show only layers[start:end]
)

# Count parameters
print("Trainable params:", sum([tf.size(v).numpy() for v in model.trainable_variables]))

# Training
history = model.fit(
    x=None,                    # Input: array, dataset, dict, generator
    y=None,                    # Labels (not needed if x is a dataset that includes them)
    batch_size=32,             # Samples per gradient update (ignored if x is dataset)
    epochs=50,
    verbose='auto',            # 0=silent, 1=progress bar, 2=single line per epoch
    callbacks=[cb_early, cb_checkpoint, cb_reduce_lr, cb_tb],
    validation_split=0.2,      # Fraction of training data used for validation
    validation_data=None,      # Explicit (x_val, y_val) or dataset
    shuffle=True,              # Shuffle training data each epoch
    class_weight={0: 1.0, 1: 10.0},  # Per-class loss weights for imbalanced data
    sample_weight=None,        # Per-sample weights array
    initial_epoch=0,           # Resume training from this epoch
    steps_per_epoch=None,      # Override steps per epoch (for generators)
    validation_steps=None,     # Validation steps per epoch
    validation_batch_size=None,
    validation_freq=1,         # Validate every N epochs
)

# Evaluation
results = model.evaluate(
    x=None,                    # Test data
    y=None,                    # Test labels
    batch_size=32,
    verbose='auto',
    sample_weight=None,
    steps=None,                # Steps for generator input
    callbacks=None,
    return_dict=True,          # Return dict {metric_name: value} vs list
)

# Prediction
preds = model.predict(
    x=None,                    # Input data
    batch_size=32,
    verbose='auto',
    steps=None,
    callbacks=None,
)


# ==============================================================================
# SECTION 11 — REGULARIZATION TECHNIQUES
# ==============================================================================

from tensorflow.keras import regularizers, constraints, initializers

# Regularizers
regularizers.L1(l1=1e-4)             # L1 norm penalty: drives weights to zero
regularizers.L2(l2=1e-4)             # L2 norm penalty: shrinks weights
regularizers.L1L2(l1=1e-5, l2=1e-4) # Combined L1 + L2 (Elastic Net)
regularizers.OrthogonalRegularizer(factor=0.01, mode='rows')  # Encourages orthogonality

# Constraints
constraints.MaxNorm(max_value=2, axis=0)         # Clip weight norm to max_value
constraints.NonNeg()                              # Force non-negative weights
constraints.UnitNorm(axis=0)                      # Normalize weights to unit norm
constraints.MinMaxNorm(min_value=0.5, max_value=2.0, rate=1.0, axis=0)

# Initializers
initializers.GlorotUniform(seed=None)   # Xavier uniform: var = 2/(fan_in+fan_out)
initializers.GlorotNormal(seed=None)    # Xavier normal variant
initializers.HeUniform(seed=None)       # He uniform: var = 2/fan_in (for ReLU)
initializers.HeNormal(seed=None)        # He normal variant
initializers.LecunUniform(seed=None)    # LeCun uniform: var = 1/fan_in (SELU)
initializers.LecunNormal(seed=None)     # LeCun normal variant
initializers.Orthogonal(gain=1.0, seed=None)  # Orthogonal matrix init (good for RNN)
initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
initializers.Constant(value=0)         # Fill with constant value
initializers.Zeros()                   # All zeros
initializers.Ones()                    # All ones
initializers.Identity(gain=1.0)        # Identity matrix (2D only)


# ==============================================================================
# SECTION 12 — ADVANCED ARCHITECTURES
# ==============================================================================

# ── 12.1  Vision Transformer (ViT) block ─────────────────────────────────────

class PatchEmbedding(layers.Layer):
    """Split image into patches and project to embedding dimension."""
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.proj = layers.Dense(embed_dim) # Linear projection of flattened patches

    def call(self, images):
        B, H, W, C = tf.shape(images)[0], images.shape[1], images.shape[2], images.shape[3]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],  # Patch size
            strides=[1, self.patch_size, self.patch_size, 1],# Non-overlapping patches
            rates=[1, 1, 1, 1],          # Dilation (1 = no dilation)
            padding='VALID',
        )
        patches = tf.reshape(patches, [B, -1, patches.shape[-1]])
        return self.proj(patches)


class TransformerBlock(layers.Layer):
    """Standard Transformer encoder block with pre-norm."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp   = keras.Sequential([
            layers.Dense(embed_dim * mlp_ratio, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        # Pre-norm architecture (more stable than post-norm)
        x = x + self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


# ── 12.2  Complete Transformer Encoder Classifier ───────────────────────────

def build_vit(
    image_size=224,            # Input image height/width (square assumed)
    patch_size=16,             # Patch size in pixels
    num_classes=1000,          # Number of output classes
    embed_dim=768,             # Token embedding dimension
    num_heads=12,              # Attention heads
    num_layers=12,             # Transformer encoder blocks
    mlp_ratio=4,               # FFN hidden size = embed_dim * mlp_ratio
    dropout=0.1,               # Dropout probability
):
    n_patches = (image_size // patch_size) ** 2
    inputs = Input(shape=(image_size, image_size, 3))

    # Patch embedding + [CLS] token + positional encoding
    x = PatchEmbedding(patch_size, embed_dim)(inputs)
    cls_token = tf.Variable(tf.zeros((1, 1, embed_dim)), trainable=True, name='cls')
    cls_tokens = tf.broadcast_to(cls_token, [tf.shape(inputs)[0], 1, embed_dim])
    x = tf.concat([cls_tokens, x], axis=1)           # Prepend CLS token

    pos_embed = tf.Variable(
        tf.random.normal((1, n_patches + 1, embed_dim), stddev=0.02),
        trainable=True, name='pos_embed')
    x = x + pos_embed                                 # Add positional embeddings

    x = layers.Dropout(dropout)(x)

    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_output = x[:, 0]                              # Extract CLS token output
    outputs = layers.Dense(num_classes)(cls_output)

    return Model(inputs, outputs, name='ViT')


# ── 12.3  Generative Adversarial Network (GAN) ──────────────────────────────

def build_generator(latent_dim=128, img_shape=(28, 28, 1)):
    """DCGAN-style generator: latent → image."""
    model = keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(img_shape[-1], 5, strides=2, padding='same',
                               use_bias=False, activation='tanh'),
    ], name='generator')
    return model


def build_discriminator(img_shape=(28, 28, 1)):
    """DCGAN-style discriminator: image → real/fake score."""
    model = keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),               # Raw logit (use BCEWithLogits for stability)
    ], name='discriminator')
    return model


class DCGAN(Model):
    """Self-contained GAN with custom train_step."""

    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator     = generator
        self.discriminator = discriminator
        self.latent_dim    = latent_dim

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super().compile()
        self.g_opt  = g_optimizer
        self.d_opt  = d_optimizer
        self.loss   = loss_fn
        self.g_loss_tracker = metrics.Mean(name='g_loss')
        self.d_loss_tracker = metrics.Mean(name='d_loss')

    def train_step(self, real_images):
        batch = tf.shape(real_images)[0]
        noise = tf.random.normal([batch, self.latent_dim])

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake = self.generator(noise, training=True)
            real_out = self.discriminator(real_images, training=True)
            fake_out = self.discriminator(fake, training=True)

            # Discriminator: real → 1, fake → 0
            d_loss = (self.loss(tf.ones_like(real_out),  real_out) +
                      self.loss(tf.zeros_like(fake_out), fake_out)) / 2

            # Generator: fool discriminator → real_out for fake images
            g_loss = self.loss(tf.ones_like(fake_out), fake_out)

        self.d_opt.apply_gradients(zip(
            d_tape.gradient(d_loss, self.discriminator.trainable_variables),
            self.discriminator.trainable_variables))
        self.g_opt.apply_gradients(zip(
            g_tape.gradient(g_loss, self.generator.trainable_variables),
            self.generator.trainable_variables))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {'g_loss': self.g_loss_tracker.result(),
                'd_loss': self.d_loss_tracker.result()}


# ==============================================================================
# SECTION 13 — MIXED PRECISION & PERFORMANCE
# ==============================================================================

from tensorflow.keras import mixed_precision

# Enable mixed precision (float16 compute, float32 variables)
policy = mixed_precision.Policy(
    'mixed_float16',           # 'float32', 'float16', 'bfloat16', 'mixed_float16', 'mixed_bfloat16'
)
mixed_precision.set_global_policy(policy)
# After setting, Dense layers auto-use float16 computation but float32 variables

# XLA compilation via tf.function
@tf.function(jit_compile=True)  # Enable XLA for this function
def compiled_step(x):
    return model(x, training=False)

# tf.function with concrete input signature (avoids retracing)
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32),
    tf.TensorSpec(shape=[None], dtype=tf.int32),
])
def train_step_fixed(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss  = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, preds))
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ==============================================================================
# SECTION 14 — SAVING & LOADING MODELS
# ==============================================================================

# ── 14.1  Keras native format (.keras) ───────────────────────────────────────
model.save(
    filepath='model.keras',    # .keras extension = new Keras v3 format (recommended)
    overwrite=True,            # Overwrite if file exists
    save_format=None,          # Auto-detected from extension
)
loaded = keras.models.load_model(
    filepath='model.keras',
    custom_objects=None,       # Dict of custom class name → class for custom layers
    compile=True,              # Restore compilation config (optimizer, loss, metrics)
    safe_mode=True,            # Restrict loading to safe objects only
)

# ── 14.2  SavedModel format (for TF Serving / TFLite) ───────────────────────
model.save('saved_model_dir')  # Directory format for TF ecosystem interop

loaded_saved = tf.saved_model.load('saved_model_dir')
infer = loaded_saved.signatures['serving_default']  # Get inference function
output = infer(input=tf.constant(np.random.randn(1, 784).astype(np.float32)))

# ── 14.3  Weights only ───────────────────────────────────────────────────────
model.save_weights('weights.weights.h5')   # Save only weights (not architecture)
model.load_weights(
    filepath='weights.weights.h5',
    by_name=False,             # Match by position (False) or layer name (True)
    skip_mismatch=False,       # Skip layers with mismatched shapes
    options=None,              # tf.train.CheckpointOptions
)

# ── 14.4  TensorFlow Checkpoint (fine-grained control) ───────────────────────
ckpt = tf.train.Checkpoint(
    model=model,               # Objects to checkpoint (named keyword args)
    optimizer=model.optimizer,
    step=tf.Variable(0),       # Track global step
)
manager = tf.train.CheckpointManager(
    checkpoint=ckpt,
    directory='./checkpoints', # Directory to save checkpoint files
    max_to_keep=3,             # Number of recent checkpoints to keep
    keep_checkpoint_every_n_hours=None,
    checkpoint_name='ckpt',    # Prefix for checkpoint files
    checkpoint_interval=None,  # Min time between checkpoints in seconds
    init_fn=None,              # Function called if no checkpoint found
)
manager.save()                 # Save current checkpoint
ckpt.restore(manager.latest_checkpoint)  # Restore latest checkpoint

# ── 14.5  Export for serving ─────────────────────────────────────────────────
@tf.function(input_signature=[tf.TensorSpec([None, 784], tf.float32)])
def serve(x):
    """Serving function with preprocessing baked in."""
    x = x / 255.0              # Normalize inside the graph
    return {'predictions': model(x)}

tf.saved_model.save(
    obj=model,                 # Model or module to save
    export_dir='serving_model',
    signatures=serve,          # Serving signature(s)
    options=None,              # tf.saved_model.SaveOptions
)


# ==============================================================================
# SECTION 15 — DISTRIBUTED TRAINING
# ==============================================================================

# ── 15.1  MirroredStrategy — single machine, multiple GPUs ───────────────────

strategy = tf.distribute.MirroredStrategy(
    devices=None,              # List of device strings; None = use all GPUs
    cross_device_ops=None,     # Communication ops; None = auto-select best
)
print("Replicas:", strategy.num_replicas_in_sync)

with strategy.scope():
    # All variables created here are mirrored across GPUs
    dist_model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax'),
    ])
    dist_model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

# ── 15.2  TPUStrategy ────────────────────────────────────────────────────────
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)

# ── 15.3  MultiWorkerMirroredStrategy — multi-machine ───────────────────────
# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {'worker': ['host1:port', 'host2:port']},
#     'task':    {'type': 'worker', 'index': 0}
# })
# strategy = tf.distribute.MultiWorkerMirroredStrategy()

# ── 15.4  Custom distributed training step ───────────────────────────────────

def distributed_train_step(strategy, model, optimizer, loss_fn, x, y):
    """Run one step across all replicas."""
    per_replica_losses = strategy.run(
        fn=lambda x, y: _step(model, optimizer, loss_fn, x, y),
        args=(x, y),           # Arguments forwarded to fn on each replica
    )
    return strategy.reduce(
        reduce_op=tf.distribute.ReduceOp.SUM,  # 'SUM' or 'MEAN'
        value=per_replica_losses,
        axis=None,
    )

def _step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss  = loss_fn(y, preds)
        loss  = loss / strategy.num_replicas_in_sync  # Scale for correct averaging
    optimizer.apply_gradients(
        zip(tape.gradient(loss, model.trainable_variables),
            model.trainable_variables))
    return loss


# ==============================================================================
# SECTION 16 — TF FUNCTIONS & GRAPH EXECUTION
# ==============================================================================

# ── 16.1  tf.function — trace and compile Python to graph ───────────────────

@tf.function(
    input_signature=None,      # List of TensorSpec; None = trace on first call
    autograph=True,            # Convert Python control flow (if/for) to TF ops
    jit_compile=False,         # Enable XLA acceleration
    reduce_retracing=False,    # Avoid retracing for slightly different inputs
    experimental_implements=None,
    experimental_autograph_options=None,
    experimental_relax_shapes=None,  # Deprecated; use reduce_retracing
)
def my_func(x, training=False):
    return model(x, training=training)

# Inspect traces
print(my_func.get_concrete_function(
    tf.TensorSpec([None, 784], tf.float32)).graph.as_graph_def())

# ── 16.2  Control flow in graphs ─────────────────────────────────────────────

@tf.function
def dynamic_op(x):
    # tf.cond — graph-compatible if/else
    return tf.cond(
        pred=tf.reduce_mean(x) > 0,  # Condition tensor
        true_fn=lambda: x * 2,       # Executed when condition is True
        false_fn=lambda: x * -1,     # Executed when condition is False
        name=None,
    )

@tf.function
def loop_op(x):
    # tf.while_loop — graph-compatible while loop
    i = tf.constant(0)
    cond   = lambda i, x: i < 5
    body   = lambda i, x: (i + 1, x + 1.0)
    _, result = tf.while_loop(
        cond=cond,                 # Loop condition function
        body=body,                 # Loop body function
        loop_vars=[i, x],          # Variables to track across iterations
        shape_invariants=None,     # Shape specs for loop_vars that change shape
        parallel_iterations=10,    # Max parallel iterations
        back_prop=True,            # Enable gradient computation through loop
        swap_memory=False,         # Swap tensors to CPU when not needed (reduces GPU memory)
        maximum_iterations=None,   # Hard cap on iterations
        name=None,
    )
    return result


# ==============================================================================
# SECTION 17 — TF.KERAS ADVANCED PATTERNS
# ==============================================================================

# ── 17.1  Multi-input multi-output model ─────────────────────────────────────

def build_multi_io_model():
    # Text branch
    text_in = Input(shape=(100,), name='text')
    x1 = layers.Embedding(10000, 64)(text_in)
    x1 = layers.LSTM(64)(x1)

    # Image branch
    img_in = Input(shape=(32, 32, 3), name='image')
    x2 = layers.Conv2D(32, 3, activation='relu')(img_in)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Fusion
    merged = layers.Concatenate()([x1, x2])
    merged = layers.Dense(64, activation='relu')(merged)

    # Multiple outputs
    class_out = layers.Dense(10, activation='softmax', name='class')(merged)
    score_out = layers.Dense(1, name='score')(merged)

    return Model([text_in, img_in], [class_out, score_out])


# ── 17.2  Transfer learning & fine-tuning ────────────────────────────────────

base_model = keras.applications.EfficientNetV2S(
    include_top=False,             # Exclude the ImageNet classification head
    weights='imagenet',            # Pre-trained weights source
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling='avg',                 # Global pooling: None, 'avg', 'max'
    classes=1000,                  # Only used if include_top=True
    include_preprocessing=True,    # Include built-in preprocessing layer
    classifier_activation='softmax',
)

# Freeze backbone for initial training
base_model.trainable = False       # Freeze all layers at once

# Build new head
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # training=False keeps BN in inference mode
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation='softmax')(x)
transfer_model = Model(inputs, outputs)

# Phase 1: Train head only
transfer_model.compile(optimizer=optimizers.Adam(1e-3),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# Phase 2: Unfreeze and fine-tune with low lr
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False
transfer_model.compile(optimizer=optimizers.Adam(1e-5),  # Much lower lr
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])


# ── 17.3  Knowledge Distillation ─────────────────────────────────────────────

class Distiller(Model):
    """Train a small student model to mimic a large teacher model."""

    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn,
                alpha=0.1,          # Weight for hard label loss (ground truth)
                temperature=5,      # Softens probability distributions for soft targets
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn     = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha       = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_preds = self.teacher(x, training=False)  # Teacher in inference mode

        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)
            # Hard loss: student vs true labels
            student_loss  = self.student_loss_fn(y, student_preds)
            # Soft loss: student vs teacher (temperature-scaled)
            distill_loss  = self.distillation_loss_fn(
                tf.nn.softmax(teacher_preds / self.temperature, axis=1),
                tf.nn.softmax(student_preds / self.temperature, axis=1),
            ) * self.temperature ** 2   # Scale back to original magnitude
            # Combined loss
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.student.trainable_variables))
        for m in self.metrics:
            m.update_state(y, student_preds)
        return {'distill_loss': distill_loss, 'student_loss': student_loss}


# ==============================================================================
# SECTION 18 — TF.IMAGE & AUGMENTATION
# ==============================================================================

img = tf.random.uniform((1, 224, 224, 3))

# Geometric
tf.image.flip_left_right(img)                              # Horizontal flip
tf.image.flip_up_down(img)                                 # Vertical flip
tf.image.rot90(img, k=1)                                   # 90-degree rotation
tf.image.random_flip_left_right(img, seed=None)
tf.image.random_flip_up_down(img, seed=None)

tf.image.central_crop(
    image=img,
    central_fraction=0.8,      # Fraction of size to keep in center
)
tf.image.crop_to_bounding_box(
    image=img,
    offset_height=10,          # Vertical offset of top-left corner
    offset_width=10,           # Horizontal offset of top-left corner
    target_height=200,         # Height of cropped region
    target_width=200,
)
tf.image.random_crop(
    value=img,
    size=(1, 200, 200, 3),     # Shape of cropped output
    seed=None,
    name=None,
)

# Color
tf.image.random_brightness(img, max_delta=0.3, seed=None)  # Add random brightness
tf.image.random_contrast(img, lower=0.7, upper=1.3, seed=None)
tf.image.random_saturation(img, lower=0.7, upper=1.3, seed=None)
tf.image.random_hue(img, max_delta=0.1, seed=None)         # Shift hue randomly
tf.image.adjust_gamma(img, gamma=1.2, gain=1)              # Gamma correction
tf.image.per_image_standardization(img[0])                 # Zero-mean, unit-variance per image

# Resize
tf.image.resize(img, size=(128, 128), method='bilinear',
                preserve_aspect_ratio=False, antialias=False)
tf.image.resize_with_crop_or_pad(img, target_height=200, target_width=200)

# Keras augmentation layers (applied only during training)
aug_pipeline = keras.Sequential([
    layers.RandomFlip('horizontal', seed=42),       # Random horizontal flip
    layers.RandomRotation(0.1, seed=42),            # Rotate ±10% of 2π
    layers.RandomZoom(0.2, seed=42),                # Zoom in/out up to 20%
    layers.RandomTranslation(0.1, 0.1, seed=42),   # Translate ±10%
    layers.RandomContrast(0.2, seed=42),            # Adjust contrast randomly
    layers.RandomBrightness(0.2, seed=42),          # Adjust brightness randomly
    layers.RandomCrop(height=200, width=200),       # Random crop to given size
    layers.Rescaling(scale=1./255, offset=0),       # Scale pixels to [0,1]
])


# ==============================================================================
# SECTION 19 — TENSORBOARD & PROFILING
# ==============================================================================

# Create file writers
train_writer = tf.summary.create_file_writer('./logs/train')
val_writer   = tf.summary.create_file_writer('./logs/val')

step = 0
with train_writer.as_default():
    tf.summary.scalar(
        name='loss',               # Metric name shown in TensorBoard
        data=0.42,                 # Scalar value
        step=step,                 # Global step counter
        description=None,          # Optional description string
    )
    tf.summary.histogram(
        name='weights',            # Name for the histogram
        data=tf.random.normal((100,)),  # Tensor to histogram
        step=step,
        buckets=None,              # Number of buckets (None=auto)
        description=None,
    )
    tf.summary.image(
        name='sample',             # Name for the image summary
        data=tf.random.uniform((4, 28, 28, 1)),  # (batch, H, W, C)
        step=step,
        max_outputs=4,             # Maximum images to display per step
        description=None,
    )

# Profiling — captures performance data for TF Profiler in TensorBoard
tf.profiler.experimental.start(
    logdir='./logs/profile',       # Directory for profile data
    options=tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2,       # Host tracing detail: 0=off, 1=basic, 2=detailed
        python_tracer_level=0,     # Python-level tracing (0=off, 1=on; expensive)
        device_tracer_level=1,     # Device (GPU) tracing detail
        delay_ms=0,                # Delay before profiling starts
    )
)
# ... run some steps ...
tf.profiler.experimental.stop()


# ==============================================================================
# SECTION 20 — MODEL CONVERSION & DEPLOYMENT
# ==============================================================================

# ── 20.1  TensorFlow Lite (mobile/edge) ─────────────────────────────────────

converter = tf.lite.TFLiteConverter.from_keras_model(
    model=model,               # Keras model to convert
)

# Optimization flags
converter.optimizations = [
    tf.lite.Optimize.DEFAULT,  # Default = post-training quantization
]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,     # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS,       # Fall back to TF ops for unsupported ops
]
converter.target_spec.supported_types = [tf.float16]  # float16 quantization

# Full integer quantization (INT8 — for microcontrollers)
def representative_dataset():
    for _ in range(100):
        data = np.random.randn(1, 784).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset  # Calibration data
converter.inference_input_type  = tf.int8   # Input dtype for INT8 model
converter.inference_output_type = tf.int8   # Output dtype for INT8 model

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Run TFLite inference
interpreter = tf.lite.Interpreter(
    model_path='model.tflite',  # Path to .tflite file
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN,
    experimental_preserve_all_tensors=False,  # Keep intermediate tensors for debugging
)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()    # Input tensor metadata
output_details = interpreter.get_output_details()   # Output tensor metadata

interpreter.set_tensor(input_details[0]['index'],
                        np.random.randn(1, 784).astype(np.float32))
interpreter.invoke()            # Run inference
output = interpreter.get_tensor(output_details[0]['index'])

# ── 20.2  TF Serving signature ───────────────────────────────────────────────

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32, name='dense_input')
])
def serving_fn(x):
    return {
        'predictions': model(x, training=False),
        'class_ids':   tf.argmax(model(x, training=False), axis=1),
    }

tf.saved_model.save(model, 'serving_model',
                    signatures={'serving_default': serving_fn})
# Deploy: docker run -p 8501:8501 -v ./serving_model:/models/mymodel
#         -e MODEL_NAME=mymodel tensorflow/serving

# ── 20.3  TensorFlow.js conversion (in terminal) ────────────────────────────
# tensorflowjs_converter --input_format=tf_saved_model \
#     --output_format=tfjs_graph_model \
#     saved_model_dir/ tfjs_model/


# ==============================================================================
# SECTION 21 — TENSORFLOW EXTRAS & UTILITIES
# ==============================================================================

# ── 21.1  TF datasets (TFDS) ─────────────────────────────────────────────────

import tensorflow_datasets as tfds

ds_train, ds_info = tfds.load(
    name='mnist',              # Dataset name from TFDS catalog
    split='train',             # Split: 'train', 'test', 'train[:80%]', etc.
    as_supervised=True,        # Return (input, label) tuple
    with_info=True,            # Also return DatasetInfo object
    data_dir=None,             # Cache directory (None=default ~/tensorflow_datasets)
    download=True,             # Download if not present locally
    shuffle_files=True,        # Shuffle shards when loading
    batch_size=None,           # Pre-batch; None=unbatched
    decoders=None,             # Custom decoder dict for features
    read_config=None,          # tfds.ReadConfig for fine-grained control
)
print(ds_info.features)        # Feature descriptions
print(ds_info.splits)          # Available split information
print(ds_info.num_examples)    # Total example count

# ── 21.2  tf.io — reading/writing data ───────────────────────────────────────

# Writing TFRecord
def serialize_example(image_bytes, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()   # Serialize to byte string

writer = tf.io.TFRecordWriter(
    path='output.tfrecord',    # Output file path
    options=None,              # Compression: None, 'ZLIB', 'GZIP', or TFRecordOptions
)
# writer.write(serialize_example(...))
writer.close()

# Reading
feature_desc = {
    'image': tf.io.FixedLenFeature([], tf.string),     # Fixed-length scalar feature
    'label': tf.io.FixedLenFeature([], tf.int64),
}
# For variable-length features:
# tf.io.VarLenFeature(dtype)             → Returns SparseTensor
# tf.io.RaggedFeature(dtype)             → Returns RaggedTensor
# tf.io.FixedLenSequenceFeature(shape)   → For sequence examples

# Decoding images
tf.io.decode_jpeg(contents=b'', channels=3, ratio=1, fancy_upscaling=True,
                  try_recover_truncated=False, acceptable_fraction=1,
                  dct_method='', name=None)
tf.io.decode_png(contents=b'', channels=0,  # 0=auto-detect channels
                 dtype=tf.uint8, name=None)
tf.io.decode_image(contents=b'', channels=None, dtype=tf.uint8,
                   expand_animations=True, name=None)

# ── 21.3  tf.strings — text operations ───────────────────────────────────────

tf.strings.split(input='hello world', sep=' ')     # Split string by separator
tf.strings.lower(input='Hello World')              # Lowercase
tf.strings.upper(input='hello')                    # Uppercase
tf.strings.strip(input='  hello  ')                # Strip whitespace
tf.strings.join(inputs=['a', 'b'], separator='-')  # Join strings
tf.strings.regex_replace(input='a1b2', pattern=r'\d', rewrite='')  # Regex replace
tf.strings.unicode_decode('hello', input_encoding='UTF-8')  # Decode to codepoints
tf.strings.bytes_split(input='hello')              # Split to individual bytes

# ── 21.4  tf.ragged — variable-length sequences ──────────────────────────────

ragged = tf.ragged.constant(
    value=[[1, 2, 3], [4, 5], [6]],  # Lists of different lengths
    dtype=None,
    ragged_rank=1,             # Number of ragged dimensions
    row_splits_dtype=tf.int64,
    name=None,
)
print(ragged.shape)            # (3, None) — None indicates ragged dimension

# ── 21.5  tf.sparse — sparse tensors ─────────────────────────────────────────

sparse = tf.SparseTensor(
    indices=[[0, 1], [1, 2]],  # Indices of non-zero values
    values=[3.0, 5.0],         # Non-zero values
    dense_shape=[3, 4],        # Full tensor shape
)
dense = tf.sparse.to_dense(
    sp_input=sparse,
    default_value=0,           # Value for unspecified indices
    validate_indices=True,
    name=None,
)

# ── 21.6  tf.signal — signal processing ─────────────────────────────────────

waveform = tf.random.normal((16000,))   # 1-second audio at 16kHz

stft = tf.signal.stft(
    signals=waveform,          # Input signal tensor
    frame_length=512,          # Length of each frame in samples
    frame_step=256,            # Step between frames (50% overlap here)
    fft_unique_bins=None,      # Unique FFT bins (None = frame_length//2 + 1)
    window_fn=tf.signal.hann_window,  # Window function applied to each frame
    pad_end=False,             # Pad signal end to fit last frame
    name=None,
)

mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=80,           # Number of mel frequency bins
    num_spectrogram_bins=stft.shape[-1],  # Number of spectrogram bins
    sample_rate=16000,         # Audio sample rate in Hz
    lower_edge_hertz=80.0,     # Lowest frequency for mel filter bank
    upper_edge_hertz=7600.0,   # Highest frequency for mel filter bank
    dtype=tf.float32,
    name=None,
)

spectrogram  = tf.abs(stft)
mel_spectrum = tf.matmul(spectrogram, mel_filterbank)
log_mel      = tf.math.log(mel_spectrum + 1e-6)    # Log-mel spectrogram


# ==============================================================================
# END OF TENSORFLOW COMPLETE GUIDE
# Topics covered:
#   1  Tensors — creation, types, shapes, indexing, math
#   2  Variables & Automatic Differentiation (GradientTape, Jacobian)
#   3  tf.data pipeline — creation, transformations, production patterns
#   4  Building models — Sequential, Functional, Subclassing
#   5  All Keras layers with every parameter
#   6  Loss functions (all built-in + custom focal loss)
#   7  Optimizers + Learning rate schedulers (all built-in)
#   8  Metrics (all built-in + custom)
#   9  Callbacks (all built-in + custom gradient monitor)
#  10  Model compilation, training, evaluation, prediction
#  11  Regularization, constraints, initializers
#  12  Advanced architectures — ViT, DCGAN, U-Net
#  13  Mixed precision & XLA / tf.function
#  14  Saving & loading — .keras, SavedModel, weights, checkpoints, TF Serving
#  15  Distributed training — MirroredStrategy, TPU, MultiWorker
#  16  Graph execution, tf.cond, tf.while_loop
#  17  Transfer learning, fine-tuning, knowledge distillation
#  18  Image augmentation (tf.image + Keras layers)
#  19  TensorBoard & profiling
#  20  TFLite conversion & deployment, TF.js
#  21  TFDS, TFRecord I/O, strings, ragged tensors, sparse tensors, signals
# ==============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          COMPLETE NEURAL NETWORK WORKFLOW — TENSORFLOW MASTER GUIDE         ║
║                                                                              ║
║  Every phase of building, training, evaluating, and deploying a neural      ║
║  network is covered below with:                                              ║
║    ► What it does          ► Why it matters (importance)                    ║
║    ► Rules to follow       ► Requirements / prerequisites                   ║
║    ► TensorFlow API used   ► Function signatures explained                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

WORKFLOW PHASES (in execution order):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  0.  Environment & Hardware Setup                                        │
  │  1.  Data Collection Concepts                                            │
  │  2.  Data Preprocessing & Feature Engineering                           │
  │  3.  tf.data Pipeline Construction                                       │
  │  4.  Model Architecture Design                                           │
  │       4a. Sequential API                                                 │
  │       4b. Functional API                                                 │
  │       4c. Subclassing API                                                │
  │       4d. Custom Layers                                                  │
  │  5.  Weight Initialization                                               │
  │  6.  Activation Functions                                                │
  │  7.  Loss Functions                                                      │
  │  8.  Optimizers                                                          │
  │  9.  Learning Rate Schedules                                             │
  │  10. Regularization Techniques                                           │
  │  11. Normalization Layers                                                │
  │  12. Model Compilation                                                   │
  │  13. Callbacks                                                           │
  │  14. Training Loop (fit API)                                             │
  │  15. Custom Training Loop (GradientTape)                                │
  │  16. Evaluation & Metrics                                                │
  │  17. Prediction & Inference                                              │
  │  18. Debugging & Visualization                                           │
  │  19. Model Saving & Loading                                              │
  │  20. Export & Deployment                                                 │
  └─────────────────────────────────────────────────────────────────────────┘
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 0 ── ENVIRONMENT & HARDWARE SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE  : Configure TensorFlow runtime, GPU memory, and reproducibility.
IMPORTANCE: Wrong hardware config causes OOM crashes or silent CPU fallback.
            Seed fixing ensures reproducible experiments.

RULES:
  ✦ Always configure GPU memory growth BEFORE importing Keras or building models.
  ✦ Set all seeds (Python, NumPy, TensorFlow) for reproducibility.
  ✦ Never hard-code device strings — let TF auto-place ops.

REQUIREMENTS:
  • TensorFlow ≥ 2.x
  • pip install tensorflow  (CPU)
  • pip install tensorflow[and-cuda]  (GPU — Linux)
"""

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,         # All layer types (Dense, Conv2D, LSTM, …)
    Model,          # Base class for subclassed models
    Input,          # Symbolic input tensor for Functional API
    regularizers,   # L1, L2, L1L2 weight penalties
    initializers,   # Glorot, He, Orthogonal, … weight initializers
    constraints,    # MaxNorm, NonNeg, … weight constraints
    optimizers,     # Adam, SGD, AdamW, RMSprop, …
    losses,         # All built-in loss functions
    metrics,        # Accuracy, AUC, Precision, Recall, …
    callbacks,      # EarlyStopping, ModelCheckpoint, TensorBoard, …
    mixed_precision # FP16 / BF16 training policies
)

# ── tf.random.set_seed(seed) ──────────────────────────────────────────────────
# WHAT    : Sets the global TF random seed for all TF ops.
# WHY     : Reproducible weight init, data shuffles, and augmentation.
# RULE    : Must be called at the very start before any TF op.
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ── tf.config.list_physical_devices(device_type) ─────────────────────────────
# WHAT    : Returns a list of available physical devices (CPU/GPU/TPU).
# WHY     : Verify GPU is visible before training.
# RETURNS : List[PhysicalDevice]
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs found: {gpus}")

# ── tf.config.experimental.set_memory_growth(device, enable) ─────────────────
# WHAT    : Allows GPU memory to grow incrementally instead of grabbing all VRAM.
# WHY     : Prevents OOM when sharing GPU with other processes.
# RULE    : Call BEFORE any TF computation. Cannot be changed after first op.
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── tf.config.set_logical_device_configuration ───────────────────────────────
# WHAT    : Split one physical GPU into multiple logical GPUs (for testing
#            multi-GPU code on a single card).
# WHY     : Useful for testing MirroredStrategy without multiple real GPUs.
# NOTE    : Only usable in testing/dev — not production.
# (Commented out to avoid conflicts with memory growth setting above)
# tf.config.set_logical_device_configuration(
#     gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
#               tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

print(f"TensorFlow version : {tf.__version__}")
print(f"Eager mode         : {tf.executing_eagerly()}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1 ── DATA COLLECTION (Conceptual)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Gather raw data that the network will learn from.
IMPORTANCE: Garbage in → garbage out. Data quality is more impactful than
             architecture choice in most real-world tasks.

RULES:
  ✦ Collect ≥10× more samples than model parameters if possible.
  ✦ Ensure class balance (or plan for imbalance handling).
  ✦ Keep a held-out test set untouched until final evaluation.
  ✦ Document data provenance (source, collection date, license).

SOURCES (TF APIs):
  tf.keras.datasets.*           → Built-in datasets (MNIST, CIFAR, IMDB…)
  tensorflow_datasets (tfds)    → 200+ curated datasets
  tf.data.experimental.CsvDataset → Stream from CSV files
  tf.io.read_file               → Read arbitrary files from disk
"""

# ── tf.keras.datasets.mnist.load_data() ──────────────────────────────────────
# WHAT    : Downloads and caches MNIST (60 000 train / 10 000 test images).
# RETURNS : Tuple of ((x_train, y_train), (x_test, y_test)) as numpy arrays.
# WHY     : Baseline dataset for demonstrating classification pipelines.
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = \
    tf.keras.datasets.mnist.load_data()

print(f"\nRaw train shape : {x_train_raw.shape}, dtype={x_train_raw.dtype}")
print(f"Raw test shape  : {x_test_raw.shape}")
print(f"Label range     : {y_train_raw.min()} – {y_train_raw.max()}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2 ── DATA PREPROCESSING & FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Transform raw data into a form the network can learn from.
IMPORTANCE: Most neural network failures are preprocessing failures.
             Proper scaling ensures gradients flow and convergence is stable.

RULES:
  ✦ Compute statistics (mean, std, min, max) on TRAINING set ONLY.
     Apply those statistics to validation and test sets.
     Reason: Using test stats is data leakage → inflated performance.
  ✦ Convert labels to integers for sparse losses, or one-hot for categorical.
  ✦ Cast inputs to float32 (default TF dtype); labels to int32 or float32.
  ✦ Always verify shapes after preprocessing.

TECHNIQUES:
  • Min-Max scaling : x' = (x - min) / (max - min)  → range [0, 1]
  • Z-score         : x' = (x - μ) / σ              → mean 0, std 1
  • One-hot encoding: tf.one_hot(labels, depth)
  • Tokenization    : tf.keras.preprocessing.text.Tokenizer

REQUIREMENTS:
  • No NaN or Inf values in input (will silently corrupt training).
  • Consistent data types across all splits.
"""

# ── tf.cast(tensor, dtype) ───────────────────────────────────────────────────
# WHAT    : Change tensor dtype without copying data when possible.
# WHY     : Nets expect float32; raw images are uint8; labels need int32.
# IMPORTANT: Always cast before arithmetic to avoid integer overflow.
x_train = tf.cast(x_train_raw, tf.float32)
x_test  = tf.cast(x_test_raw,  tf.float32)
y_train = tf.cast(y_train_raw, tf.int32)
y_test  = tf.cast(y_test_raw,  tf.int32)

# ── Manual normalization (pixel values 0–255 → 0.0–1.0) ──────────────────────
# WHAT    : Divide by 255.0 to bring pixels into [0, 1].
# WHY     : Keeps activations and gradients in a well-behaved numeric range.
# RULE    : Use training-set statistics only. Here min=0, max=255 is universal
#            for uint8 images, so we can safely use 255.0 directly.
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# ── tf.expand_dims(tensor, axis) ─────────────────────────────────────────────
# WHAT    : Insert a new dimension of size 1 at the specified axis.
# WHY     : Conv2D expects shape [batch, H, W, channels]. MNIST is [N, 28, 28].
# RULE    : axis=-1 appends channel dim at end (channels-last, TF default).
x_train = tf.expand_dims(x_train, axis=-1)   # [60000, 28, 28] → [60000, 28, 28, 1]
x_test  = tf.expand_dims(x_test,  axis=-1)

# ── tf.one_hot(indices, depth) ───────────────────────────────────────────────
# WHAT    : Convert integer class indices to one-hot vectors.
# WHY     : Required for categorical_crossentropy (non-sparse variant).
# PARAMS  : indices=label tensor, depth=number of classes
# RETURNS : Tensor of shape [..., depth] with dtype float32
y_train_oh = tf.one_hot(y_train, depth=10)   # [60000] → [60000, 10]

# ── Train / validation split ──────────────────────────────────────────────────
# RULE    : Validation set should be ~10–20% of training data.
#           Shuffle BEFORE splitting to avoid temporal bias.
VAL_SIZE   = 5000
x_val      = x_train[-VAL_SIZE:]
y_val      = y_train[-VAL_SIZE:]
x_train_s  = x_train[:-VAL_SIZE]
y_train_s  = y_train[:-VAL_SIZE]

print(f"\nPreprocessed train : {x_train_s.shape}")
print(f"Preprocessed val   : {x_val.shape}")
print(f"Preprocessed test  : {x_test.shape}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3 ── tf.data PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Efficiently stream data to the GPU, avoiding CPU bottlenecks.
IMPORTANCE: A slow data pipeline wastes GPU cycles; this is the #1 hidden
             training bottleneck in production systems.

PIPELINE ORDER (optimal):
  from_tensor_slices → cache → shuffle → map (augmentation) → batch → prefetch

RULES:
  ✦ cache() BEFORE shuffle/map for small datasets that fit in RAM.
  ✦ shuffle(buffer_size) should be ≥ dataset size for true randomness.
  ✦ prefetch(AUTOTUNE) overlaps CPU data prep with GPU compute.
  ✦ drop_remainder=True ensures constant batch size (required for XLA).
  ✦ Never shuffle the test/validation sets.

REQUIREMENTS:
  • All tensors in a batch must have the same shape (pad sequences if needed).
  • Data type must match model's expected input dtype.
"""

AUTOTUNE   = tf.data.AUTOTUNE   # TF auto-selects optimal parallelism level
BATCH_SIZE = 128                # RULE: power of 2 for GPU efficiency (32–512)

# ── tf.data.Dataset.from_tensor_slices(tensors) ──────────────────────────────
# WHAT    : Create a dataset from in-memory tensors or numpy arrays.
# WHY     : The entry point for small-medium datasets that fit in memory.
# PARAMS  : tensors → tuple/dict/single tensor; slices along axis 0.
# RETURNS : tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train_s, y_train_s))
val_ds   = tf.data.Dataset.from_tensor_slices((x_val,     y_val))
test_ds  = tf.data.Dataset.from_tensor_slices((x_test,    y_test))

# ── .cache(filename="") ───────────────────────────────────────────────────────
# WHAT    : Cache dataset in RAM (or disk if filename provided) after first epoch.
# WHY     : Eliminates repeated I/O for subsequent epochs.
# RULE    : Place before shuffle/map when dataset fits in RAM.
train_ds = train_ds.cache()

# ── .shuffle(buffer_size, seed, reshuffle_each_iteration) ─────────────────────
# WHAT    : Randomly shuffle samples within a sliding buffer.
# WHY     : Prevents the model from learning data order (temporal artifacts).
# PARAMS  : buffer_size → larger = better shuffle, more RAM used.
#            seed → fixed seed for reproducibility.
# RULE    : buffer_size = full dataset size for perfect shuffle.
train_ds = train_ds.shuffle(buffer_size=60000, seed=SEED,
                              reshuffle_each_iteration=True)

# ── .map(fn, num_parallel_calls) ─────────────────────────────────────────────
# WHAT    : Apply a function to every element of the dataset in parallel.
# WHY     : Augmentation, tokenization, or any per-sample transform.
# PARAMS  : num_parallel_calls=AUTOTUNE → TF picks optimal thread count.
# RULE    : Keep map functions tf.function-compatible (graph-friendly).
def augment(image, label):
    """
    Data augmentation applied ON-THE-FLY during training.
    IMPORTANCE: Artificially expands dataset size; prevents overfitting;
                teaches translation/illumination invariance.
    RULES:
      ✦ Only augment the training set, NEVER validation or test.
      ✦ Augmentations must preserve label semantics (don't flip '6' ↔ '9').
    """
    # tf.image.random_flip_left_right → random horizontal mirror
    # WHY: CNNs aren't inherently symmetric; flipping teaches it.
    image = tf.image.random_flip_left_right(image)

    # tf.image.random_brightness(image, max_delta) → random brightness shift
    # WHY: Simulates different lighting conditions.
    # RULE: max_delta in [0, 1]; 0.2 is a safe default.
    image = tf.image.random_brightness(image, max_delta=0.15)

    # tf.image.random_contrast(image, lower, upper) → random contrast scale
    # WHY: Simulates varying contrast (cameras, scanning, printing).
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # tf.clip_by_value(t, min, max) → clamp tensor values to a range
    # WHY: Augmentation can push pixels outside [0,1]; clamp to prevent NaN.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

# ── .batch(batch_size, drop_remainder) ───────────────────────────────────────
# WHAT    : Group consecutive elements into batches.
# WHY     : GPU parallelism requires batch processing; single samples are slow.
# PARAMS  : drop_remainder=True → discard last partial batch.
#            drop_remainder=False → last batch may be smaller.
# RULE    : drop_remainder=True when using XLA or static shapes.
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
val_ds   = val_ds.batch(BATCH_SIZE)
test_ds  = test_ds.batch(BATCH_SIZE)

# ── .prefetch(buffer_size) ────────────────────────────────────────────────────
# WHAT    : Prepare the next batch while the current one trains on GPU.
# WHY     : Hides CPU preprocessing latency → near 100% GPU utilization.
# PARAMS  : buffer_size=AUTOTUNE → TF dynamically determines optimal value.
# RULE    : Always add as the LAST pipeline step.
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ── .repeat(count) (optional) ────────────────────────────────────────────────
# WHAT    : Repeat the dataset count times (None = infinite).
# WHY     : Needed when using steps_per_epoch in model.fit().
# RULE    : Prefer epochs argument over repeat() for clarity.
# train_ds = train_ds.repeat()

print(f"\nDataset element spec: {train_ds.element_spec}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 4 ── MODEL ARCHITECTURE DESIGN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Define the computation graph (structure of layers and connections).
IMPORTANCE: Architecture determines representational capacity. Too shallow →
             underfitting. Too deep → vanishing gradients, overfitting.

THREE KERAS APIS:
  ┌──────────────┬───────────────────────────┬──────────────────────────────┐
  │ API          │ Use when                  │ Limitation                   │
  ├──────────────┼───────────────────────────┼──────────────────────────────┤
  │ Sequential   │ linear stack, beginners   │ no branching, no multi-I/O   │
  │ Functional   │ residual/multi-I/O graphs │ static graph required        │
  │ Subclassing  │ dynamic logic, research   │ less auto-tooling support    │
  └──────────────┴───────────────────────────┴──────────────────────────────┘

RULES:
  ✦ Match architecture to problem type:
      Classification → Dense/Conv2D + Softmax output
      Regression     → Dense output, NO activation
      Segmentation   → U-Net / FCN with Sigmoid output
      Sequence       → LSTM / GRU / Transformer
  ✦ First layer must know input shape (explicit or inferred from first batch).
  ✦ Final layer units = number of classes (classification) or 1 (binary/regress).
"""

# ─────────────────────────────────────────────────────────────────────────────
# 4a. SEQUENTIAL API
# ─────────────────────────────────────────────────────────────────────────────
"""
PURPOSE   : Build a linear stack of layers with one line per layer.
IMPORTANCE: Simplest API; great for prototyping and teaching.
RULE      : Use only when the model has exactly ONE input and ONE output,
             and no skip connections.
"""

# ── keras.Sequential(layers=[]) ──────────────────────────────────────────────
# WHAT    : Container that chains layers sequentially.
# PARAMS  : layers → optional list of Layer objects.
# RULE    : First layer should specify input_shape or use Input() as first element.
sequential_model = keras.Sequential(name="SequentialCNN")

# ── layers.Input(shape, dtype, name) ─────────────────────────────────────────
# WHAT    : Defines the shape and dtype of incoming data (symbolic placeholder).
# WHY     : Allows Keras to compute output shapes and validate graph before run.
# PARAMS  : shape → does NOT include batch dimension.
# RULE    : shape=(28,28,1) NOT (None,28,28,1); batch dim is implicit.
sequential_model.add(layers.Input(shape=(28, 28, 1), name="image_input"))

# ── layers.Conv2D(filters, kernel_size, strides, padding, activation) ─────────
# WHAT    : 2D convolution — slides filters over spatial dimensions.
# WHY     : Extracts local features (edges, textures, shapes) with weight sharing.
# PARAMS  :
#   filters     → number of output feature maps (depth of output tensor)
#   kernel_size → size of the sliding filter window (int or tuple)
#   strides     → step size of the filter (1=no skip, 2=halve spatial dims)
#   padding     → 'same' (output same H/W) | 'valid' (output shrinks)
#   activation  → nonlinearity applied elementwise after convolution
# RULE    : padding='same' preserves spatial dimensions for easier architecture
#            design. padding='valid' reduces spatial size by (kernel-1).
# IMPORTANCE: Convolutional layers share weights across space → far fewer
#              parameters than fully-connected for image data.
sequential_model.add(layers.Conv2D(
    filters=32, kernel_size=3, strides=1,
    padding="same", activation="relu", name="conv1"))

# ── layers.BatchNormalization(axis, momentum, epsilon) ────────────────────────
# WHAT    : Normalizes activations across the batch to mean≈0, std≈1, then
#            scales/shifts with learnable γ, β parameters.
# WHY     : Stabilizes training; acts as regularizer; allows higher LR;
#            reduces sensitivity to weight initialization.
# PARAMS  :
#   axis      → feature axis to normalize (default -1 = channel axis)
#   momentum  → exponential moving average decay for running stats (0.99)
#   epsilon   → small constant for numerical stability (1e-3 default)
# RULE    : Place AFTER linear layer, BEFORE activation (pre-activation style).
#            At inference, use running mean/variance, NOT batch statistics.
sequential_model.add(layers.BatchNormalization(momentum=0.99, name="bn1"))

# ── layers.MaxPooling2D(pool_size, strides, padding) ──────────────────────────
# WHAT    : Downsamples spatial dimensions by taking the max in each window.
# WHY     : Reduces feature map size → fewer parameters downstream;
#            introduces local translation invariance.
# PARAMS  :
#   pool_size → spatial window size (default 2×2)
#   strides   → default equals pool_size (non-overlapping windows)
# ALTERNATIVE: layers.AveragePooling2D → takes mean instead of max.
sequential_model.add(layers.MaxPooling2D(pool_size=2, name="pool1"))

# ── layers.Dropout(rate) ──────────────────────────────────────────────────────
# WHAT    : Randomly sets rate fraction of inputs to 0 at training time.
# WHY     : Regularization — forces network to learn redundant representations;
#            prevents co-adaptation of neurons; reduces overfitting.
# PARAMS  : rate → fraction to drop (0.0–0.5 typical; 0.5 = aggressive)
# RULE    : Only active during training=True. At inference, all neurons active
#            and outputs are scaled by (1-rate) automatically.
# IMPORTANT: DO NOT use after BN (conflicting normalization effects).
sequential_model.add(layers.Dropout(rate=0.25, name="drop1"))

sequential_model.add(layers.Conv2D(64, 3, padding="same", name="conv2"))
sequential_model.add(layers.BatchNormalization(name="bn2"))
sequential_model.add(layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3"))
sequential_model.add(layers.MaxPooling2D(2, name="pool2"))
sequential_model.add(layers.Dropout(0.25, name="drop2"))

# ── layers.Flatten() ──────────────────────────────────────────────────────────
# WHAT    : Collapse all spatial/feature dimensions into a single 1D vector.
# WHY     : Bridge between spatial layers (Conv) and dense layers.
# RULE    : Never flatten before spatial layers; flatten only before Dense.
# ALTERNATIVE: layers.GlobalAveragePooling2D() — better than Flatten for deep
#              nets; spatial average reduces overfitting.
sequential_model.add(layers.Flatten(name="flatten"))

# ── layers.Dense(units, activation, use_bias, kernel_regularizer) ─────────────
# WHAT    : Fully-connected layer — every input connected to every output neuron.
# WHY     : Learns global feature combinations after spatial feature extraction.
# PARAMS  :
#   units              → number of neurons (output dimension)
#   activation         → nonlinearity (relu, sigmoid, softmax, tanh, …)
#   use_bias           → add bias term (default True; set False with BN)
#   kernel_regularizer → L1/L2 penalty on weights
#   kernel_initializer → weight init strategy
#   kernel_constraint  → constraint on weights (e.g. MaxNorm)
# RULE    : Final Dense units = num_classes for classification.
#            No activation on final layer when using from_logits=True in loss.
sequential_model.add(layers.Dense(
    128, activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="dense1"))
sequential_model.add(layers.Dropout(0.5, name="drop3"))
sequential_model.add(layers.Dense(10, activation="softmax", name="output"))
# NOTE: activation="softmax" → outputs sum to 1.0 (probability distribution)
#       For binary: activation="sigmoid" (single output unit)
#       For regression: no activation (linear output)

sequential_model.summary()


# ─────────────────────────────────────────────────────────────────────────────
# 4b. FUNCTIONAL API
# ─────────────────────────────────────────────────────────────────────────────
"""
PURPOSE   : Build models as directed acyclic graphs (DAGs) of layers.
IMPORTANCE: Required for:
             • Residual / skip connections (ResNet, DenseNet)
             • Multi-input models (image + metadata)
             • Multi-output models (classification + regression head)
             • Shared layers (Siamese networks)
RULE      : Each layer call returns a tensor — thread these tensors through.
             Model(inputs=[...], outputs=[...]) finalizes the graph.
"""

def build_functional_model(input_shape=(28, 28, 1), num_classes=10):
    """Residual-style model using Functional API."""

    # Input() creates a symbolic tensor (not real data yet)
    inp = Input(shape=input_shape, name="image")

    # ── Stem ──────────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)    # Activation as separate layer (flexible)

    # ── Residual Block ─────────────────────────────────────────────────────────
    # IMPORTANCE: Skip connection allows gradients to flow through identity path;
    #              enables training of very deep networks (100+ layers).
    shortcut = x
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # ── layers.Add() ──────────────────────────────────────────────────────────
    # WHAT    : Element-wise addition of a list of tensors.
    # WHY     : Core of residual connections — adds identity shortcut to output.
    # RULE    : All tensors must have the same shape.
    # ALTERNATIVES: layers.Concatenate() → channel-wise concat (DenseNet style)
    x = layers.Add(name="residual_add")([x, shortcut])
    x = layers.Activation("relu")(x)

    # ── layers.SeparableConv2D(filters, kernel_size) ───────────────────────────
    # WHAT    : Depthwise conv (per-channel) + pointwise conv (1×1 mix).
    # WHY     : ~8–9× fewer parameters than regular Conv2D with similar accuracy.
    # USE CASE: Mobile/edge models (MobileNet architecture).
    x = layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # ── layers.GlobalAveragePooling2D() ───────────────────────────────────────
    # WHAT    : Compute the spatial average of each feature map.
    # WHY     : Replaces Flatten + Dense; fewer parameters; better generalization;
    #            model is spatially invariant (works on any input size).
    # RULE    : Preferred over Flatten for deep CNNs.
    x = layers.GlobalAveragePooling2D()(x)

    # ── Multi-output example ───────────────────────────────────────────────────
    cls_out = layers.Dense(num_classes, activation="softmax", name="class_out")(x)

    # Model(inputs, outputs) — finalizes the graph
    # WHAT    : Wraps inputs and outputs into a trainable Model object.
    # RULE    : inputs and outputs must be tensors from the same graph.
    model = Model(inputs=inp, outputs=cls_out, name="FunctionalResNet")
    return model

functional_model = build_functional_model()
functional_model.summary()


# ─────────────────────────────────────────────────────────────────────────────
# 4c. MODEL SUBCLASSING API
# ─────────────────────────────────────────────────────────────────────────────
"""
PURPOSE   : Full Python control over the forward pass.
IMPORTANCE: Enables dynamic graphs (variable computation depth, conditional
             layers, tree-structured computation) needed for research models.
RULES:
  ✦ Define all layers in __init__ (so TF can track variables).
  ✦ Implement forward pass in call(self, inputs, training=False).
  ✦ Always pass training flag to Dropout and BatchNorm layers.
  ✦ Prefer Functional API over subclassing for simpler architectures
     (Functional models are easier to save, visualize, debug).
"""

class SubclassedCNN(Model):
    """
    Subclassed model: full Python control, dynamic behaviour.
    USE CASE: Research models, dynamic computation, mixture-of-experts.
    """

    def __init__(self, num_classes=10, name="SubclassedCNN", **kwargs):
        """
        __init__: Define ALL layers here.
        WHY: TF tracks variables registered as attributes in __init__.
        RULE: Never create tf.Variable or layers inside call() — they won't
               be tracked and won't be saved correctly.
        """
        super().__init__(name=name, **kwargs)

        # ── layers.Conv2D ──────────────────────────────────────────────────
        self.conv1 = layers.Conv2D(32, 3, padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(64, 3, padding="same", use_bias=False)

        # ── layers.BatchNormalization ──────────────────────────────────────
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        # ── layers.MaxPooling2D ────────────────────────────────────────────
        self.pool = layers.MaxPooling2D(2)

        # ── layers.GlobalAveragePooling2D ──────────────────────────────────
        self.gap  = layers.GlobalAveragePooling2D()

        # ── layers.Dropout ─────────────────────────────────────────────────
        self.drop1 = layers.Dropout(0.3)
        self.drop2 = layers.Dropout(0.5)

        # ── layers.Dense ───────────────────────────────────────────────────
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(num_classes)     # logits (no activation here)

        # ── layers.Activation ──────────────────────────────────────────────
        # WHAT    : Apply activation as a standalone layer (more flexible).
        # WHY     : Separating BN → Activation allows pre-activation residuals.
        self.relu = layers.Activation("relu")

    def call(self, inputs, training=False):
        """
        call(): Define the FORWARD PASS.
        PARAMS  :
          inputs   → input tensor(s)
          training → bool/tensor; controls BN (use running stats vs batch stats)
                      and Dropout (active vs inactive).
        RULE    : Always pass training to BN and Dropout.
                   Use tf.cast(training, tf.bool) if needed for graph mode.
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)   # ← training flag critical
        x = self.relu(x)
        x = self.drop1(x, training=training) # ← training flag critical

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.pool(x)

        x = self.gap(x)
        x = self.fc1(x)
        x = self.drop2(x, training=training)
        return self.fc2(x)   # returns logits

    def get_config(self):
        """
        get_config(): Required for model serialization (save/load).
        RULE    : Return a JSON-serializable dict of all constructor arguments.
        WHY     : Allows keras.models.model_from_config() to reconstruct model.
        """
        base = super().get_config()
        base.update({"num_classes": 10})
        return base

subclassed_model = SubclassedCNN(num_classes=10)
# Build by passing a dummy input (required to initialize weights)
# WHY: Subclassed models don't know input shape until first call.
_ = subclassed_model(tf.zeros([1, 28, 28, 1]))
subclassed_model.summary()


# ─────────────────────────────────────────────────────────────────────────────
# 4d. CUSTOM LAYERS
# ─────────────────────────────────────────────────────────────────────────────
"""
PURPOSE   : Encapsulate novel operations as reusable, serializable building blocks.
IMPORTANCE: Enables research novelty; separates concerns; allows custom backprop.
RULES:
  ✦ Inherit from layers.Layer.
  ✦ Create weights in build(input_shape) — called once, lazily, on first call.
  ✦ Implement computation in call(inputs, training).
  ✦ Override get_config() for serializability.
  ✦ Use self.add_weight() (not tf.Variable) so TF tracks the weight.
"""

class SqueezeExcitation(layers.Layer):
    """
    Squeeze-and-Excitation block: channel-wise attention.
    USE CASE: Image classification (SENet, EfficientNet).
    HOW IT WORKS:
      1. Squeeze: GlobalAveragePool → scalar per channel
      2. Excitation: Two Dense layers → channel weights in [0,1]
      3. Scale: Multiply original features by channel weights
    IMPORTANCE: Teaches network which channels are most informative
                 per-input (adaptive feature recalibration).
    """

    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        """
        build(input_shape): Called ONCE on first forward pass.
        WHY: At this point TF knows the number of channels.
        RULE: Create all weights here using self.add_weight() or sub-layers.
        """
        channels = input_shape[-1]
        reduced  = max(1, channels // self.reduction_ratio)

        # self.add_weight(name, shape, initializer, trainable) ────────────────
        # WHAT    : Register a learnable tf.Variable on this layer.
        # WHY     : add_weight auto-adds to layer.trainable_variables.
        # RULE    : Prefer add_weight over tf.Variable inside layers.
        self.squeeze  = layers.GlobalAveragePooling2D()
        self.fc_down  = layers.Dense(reduced, activation="relu")
        self.fc_up    = layers.Dense(channels, activation="sigmoid")
        self.reshape  = layers.Reshape((1, 1, channels))
        super().build(input_shape)

    def call(self, inputs, training=False):
        scale = self.squeeze(inputs)           # [B, C]
        scale = self.fc_down(scale)            # [B, C//r]
        scale = self.fc_up(scale)              # [B, C]
        scale = self.reshape(scale)            # [B, 1, 1, C]
        return inputs * scale                  # channel-wise scaling

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction_ratio": self.reduction_ratio})
        return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 5 ── WEIGHT INITIALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Set initial values of weights before training begins.
IMPORTANCE: Poor initialization → vanishing/exploding gradients → training failure.
             Good initialization → faster convergence, better final accuracy.

RULES:
  ✦ NEVER initialize all weights to 0 (symmetry breaking fails — all neurons
     compute identical gradients forever).
  ✦ NEVER initialize all weights to a constant (same reason).
  ✦ Match initializer to activation:
      ReLU / LeakyReLU → He (Kaiming) Normal/Uniform
      Tanh / Sigmoid   → Glorot (Xavier) Normal/Uniform  (default)
      SELU             → LeCun Normal
  ✦ Biases: initialize to 0 (standard); small positive for ReLU (dead neuron fix).

REQUIREMENTS:
  • Initializer is passed to kernel_initializer argument of any weight layer.
"""

# ── initializers.GlorotUniform() / GlorotNormal() ────────────────────────────
# WHAT    : Xavier init — samples from distribution scaled by fan_in + fan_out.
# WHY     : Keeps variance of activations and gradients approximately equal
#            across layers for tanh/sigmoid.
# MATH    : std = sqrt(2 / (fan_in + fan_out))
glorot_init  = initializers.GlorotUniform(seed=SEED)
glorot_n     = initializers.GlorotNormal(seed=SEED)

# ── initializers.HeNormal() / HeUniform() ────────────────────────────────────
# WHAT    : Kaiming He init — scaled for ReLU's half-dead neurons.
# WHY     : Compensates for ReLU zeroing half its inputs.
# MATH    : std = sqrt(2 / fan_in)
he_init      = initializers.HeNormal(seed=SEED)

# ── initializers.Orthogonal(gain) ────────────────────────────────────────────
# WHAT    : Initializes weight matrix as a random orthogonal matrix.
# WHY     : Excellent for RNNs — preserves gradient magnitude across time steps.
# USE CASE: LSTM/GRU recurrent weights.
orth_init    = initializers.Orthogonal(gain=1.0, seed=SEED)

# ── initializers.LecunNormal() ───────────────────────────────────────────────
# WHAT    : Scaled normal; designed for SELU activation.
# WHY     : SELU self-normalizes only if weights follow LeCun Normal init.
lecun_init   = initializers.LecunNormal(seed=SEED)

# ── initializers.Constant(value) ─────────────────────────────────────────────
# WHAT    : Initialize all values to a constant.
# USE CASE: Bias initialization to a small positive value (e.g., 0.01 for ReLU).
# RULE    : Use for biases, NOT for kernels.
bias_init    = initializers.Constant(0.01)

# Example layer using explicit initialization:
init_layer = layers.Dense(
    128,
    kernel_initializer=he_init,        # weights
    bias_initializer=bias_init,        # biases
    kernel_constraint=constraints.MaxNorm(3.0),   # ||w|| ≤ 3.0
)
# constraints.MaxNorm(max_value) → rescale weight if norm exceeds max_value
# WHY     : Prevents weight explosion during long training runs.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 6 ── ACTIVATION FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Introduce non-linearity so the network can learn complex functions.
IMPORTANCE: Without activations, any deep network collapses to a single linear
             transformation (no matter how deep). Non-linearity is MANDATORY.

RULES:
  ✦ Hidden layers → use ReLU variants (GELU/SiLU for transformers).
  ✦ Output layer → activation depends on task:
      Multiclass  : softmax  (probabilities sum to 1)
      Binary      : sigmoid  (single probability in [0,1])
      Regression  : None (linear)
      Multilabel  : sigmoid  (independent probabilities per class)
  ✦ Avoid sigmoid/tanh in deep hidden layers → vanishing gradient problem.
  ✦ ReLU can "die" (always output 0); use LeakyReLU or ELU as safer defaults.

REQUIREMENTS:
  • Activation output range must be compatible with the loss function.
    (softmax+sparse_categorical_crossentropy is fine; softmax+MSE is not ideal)
"""

# ── layers.Activation("relu") ────────────────────────────────────────────────
# f(x) = max(0, x)
# PROS: Fast, sparse, no vanishing gradient for positive inputs.
# CONS: Dead ReLU (neurons stuck at 0 forever if negative).
relu_layer = layers.Activation("relu")

# ── layers.LeakyReLU(negative_slope) ─────────────────────────────────────────
# f(x) = x if x>0 else alpha*x  (alpha=0.01–0.3)
# PROS: Prevents dead neurons; small gradient for negatives.
# USE CASE: GANs (discriminator training) where dead ReLU is common.
leaky = layers.LeakyReLU(negative_slope=0.2)

# ── layers.ELU(alpha) ────────────────────────────────────────────────────────
# f(x) = x if x>0 else alpha*(exp(x)-1)
# PROS: Smooth; negative outputs help push mean activations toward 0.
# USE CASE: Deep networks where training stability is a concern.
elu_layer = layers.ELU(alpha=1.0)

# ── layers.Activation("gelu") ────────────────────────────────────────────────
# f(x) = x * Φ(x)  where Φ = Gaussian CDF
# PROS: Smooth; empirically outperforms ReLU on NLP/Transformer tasks.
# USE CASE: BERT, GPT, ViT — all use GELU.
gelu_layer = layers.Activation("gelu")

# ── layers.Activation("swish") ───────────────────────────────────────────────
# f(x) = x * sigmoid(x)  (also called SiLU)
# PROS: Smooth, non-monotonic; outperforms ReLU on many benchmarks.
# USE CASE: EfficientNet, MobileNetV3.
swish_layer = layers.Activation("swish")

# ── layers.Activation("softmax") ─────────────────────────────────────────────
# f(x_i) = exp(x_i) / sum(exp(x_j))
# RULE    : Output layer ONLY for multiclass classification.
# IMPORTANT: Use from_logits=True in loss if softmax NOT in model (faster/stable).
softmax_layer = layers.Activation("softmax")

# ── layers.Activation("sigmoid") ─────────────────────────────────────────────
# f(x) = 1 / (1 + exp(-x))  → [0, 1]
# USE CASE: Binary classification output; attention gates; VAE latent means.
sigmoid_layer = layers.Activation("sigmoid")

# ── layers.Activation("tanh") ────────────────────────────────────────────────
# f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)) → [-1, 1]
# USE CASE: LSTM/GRU cell states; output layer when output ∈ (-1,1).
tanh_layer = layers.Activation("tanh")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 7 ── LOSS FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Quantify how wrong the model's predictions are.
IMPORTANCE: The loss function defines what "correct" means.
             Wrong loss → model optimizes the wrong objective.

RULES:
  ✦ Match loss to task:
      Multiclass (integer labels) → SparseCategoricalCrossentropy
      Multiclass (one-hot labels) → CategoricalCrossentropy
      Binary (0/1 labels)        → BinaryCrossentropy
      Regression                 → MeanSquaredError or MeanAbsoluteError
      Ranking / contrastive      → custom loss
  ✦ from_logits=True (when model outputs raw logits, no softmax/sigmoid):
      PREFERRED — numerically more stable (uses log-sum-exp trick internally).
      from_logits=False: use when model outputs probabilities (softmax applied).
  ✦ reduction='sum_over_batch_size' (default): averages loss across batch.
     Use 'none' to get per-sample losses (e.g., for importance weighting).

REQUIREMENTS:
  • y_true dtype must match loss expectation:
      SparseCategorical → int32/int64 labels
      Categorical       → float32 one-hot vectors
      Binary            → float32 0.0 or 1.0
"""

# ── losses.SparseCategoricalCrossentropy(from_logits, reduction) ──────────────
# WHAT    : Cross-entropy for multiclass with INTEGER class labels.
# WHY     : Avoids explicit one-hot encoding; memory-efficient.
# MATH    : -log(p_true_class) where p is softmax probability.
loss_scc = losses.SparseCategoricalCrossentropy(
    from_logits=False,   # True if model has no softmax
    reduction="sum_over_batch_size",
    name="sparse_ce")

# ── losses.CategoricalCrossentropy(from_logits, label_smoothing) ──────────────
# WHAT    : Cross-entropy for multiclass with ONE-HOT labels.
# EXTRA   : label_smoothing → replace 0→ε and 1→(1-ε); prevents overconfidence.
# USE CASE: When your labels are already one-hot (e.g., from one_hot()).
loss_cc = losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0.1,    # WHY: Prevents model from being 100% confident.
    name="categorical_ce")

# ── losses.BinaryCrossentropy(from_logits) ────────────────────────────────────
# WHAT    : Cross-entropy for binary classification.
# MATH    : -[y*log(p) + (1-y)*log(1-p)]
# USE CASE: Spam/not-spam, disease/healthy, fraud/legitimate.
loss_bce = losses.BinaryCrossentropy(from_logits=False, name="binary_ce")

# ── losses.MeanSquaredError() ────────────────────────────────────────────────
# WHAT    : Mean of squared differences between predictions and targets.
# MATH    : (1/N) * Σ(y_pred - y_true)²
# PROS    : Differentiable everywhere; penalizes large errors heavily.
# CONS    : Sensitive to outliers (squared amplifies them).
loss_mse = losses.MeanSquaredError(name="mse")

# ── losses.MeanAbsoluteError() ───────────────────────────────────────────────
# WHAT    : Mean of absolute differences.
# MATH    : (1/N) * Σ|y_pred - y_true|
# PROS    : Robust to outliers.
# CONS    : Not differentiable at 0; gradient is constant (no scaling by error).
loss_mae = losses.MeanAbsoluteError(name="mae")

# ── losses.Huber(delta) ───────────────────────────────────────────────────────
# WHAT    : MSE for small errors, MAE for large errors (hybrid).
# WHY     : Gets MSE smooth gradient near 0 AND MAE outlier robustness.
# USE CASE: Regression with outliers (bounding box regression in object detection).
loss_hub = losses.Huber(delta=1.0, name="huber")

# ── losses.KLDivergence() ────────────────────────────────────────────────────
# WHAT    : KL divergence between two probability distributions.
# MATH    : Σ p * log(p/q)
# USE CASE: VAE (regularization term); knowledge distillation (soft targets).
loss_kl  = losses.KLDivergence(name="kl_div")

# ── Custom loss function ──────────────────────────────────────────────────────
class FocalLoss(losses.Loss):
    """
    Focal Loss: addresses class imbalance by down-weighting easy examples.
    MATH     : FL(p) = -alpha * (1-p)^gamma * log(p)
    USE CASE : Object detection (RetinaNet), severe class imbalance.
    IMPORTANCE: Cross-entropy treats all samples equally; focal loss focuses
                 training on hard misclassified examples.
    PARAMS   :
      gamma → focusing parameter (0 = cross-entropy; 2 = typical focal)
      alpha → class balance weight (0.25 typical)
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce    = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * bce, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 8 ── OPTIMIZERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Update model weights using computed gradients to minimize loss.
IMPORTANCE: The optimizer determines HOW FAST and WHETHER the model converges.
             Wrong optimizer or learning rate → no convergence or divergence.

RULES:
  ✦ Adam is the safe default for most tasks.
  ✦ SGD with momentum is often better for CNNs with proper LR schedule.
  ✦ Learning rate is the single most important hyperparameter.
  ✦ Too high LR → loss oscillates / diverges.
     Too low LR → very slow convergence or stuck in local minima.
  ✦ Weight decay (L2 regularization on optimizer) ≠ kernel_regularizer:
      AdamW applies weight decay CORRECTLY (decoupled from gradient scaling).
  ✦ Gradient clipping prevents exploding gradients (essential for RNNs).

REQUIREMENTS:
  • Optimizer must be passed to model.compile() or used with GradientTape.
  • Learning rate must be a positive float or a LearningRateSchedule object.
"""

# ── optimizers.Adam(learning_rate, beta_1, beta_2, epsilon) ───────────────────
# WHAT    : Adaptive Moment Estimation — per-parameter adaptive learning rates.
# HOW     : Maintains exponential moving averages of gradients (m) and
#            squared gradients (v). Updates: θ -= lr * m / (sqrt(v) + ε).
# PARAMS  :
#   beta_1   → decay for first moment (gradient momentum); default 0.9
#   beta_2   → decay for second moment (squared gradient); default 0.999
#   epsilon  → numerical stability constant; default 1e-7
# PROS    : Works well without LR tuning; handles sparse gradients.
# CONS    : Can generalize worse than SGD on some CV tasks.
opt_adam = optimizers.Adam(learning_rate=1e-3, beta_1=0.9,
                            beta_2=0.999, epsilon=1e-7)

# ── optimizers.AdamW(learning_rate, weight_decay) ────────────────────────────
# WHAT    : Adam with DECOUPLED weight decay (L2 regularization done right).
# WHY     : Regular Adam+L2 doesn't properly decouple weight decay from
#            adaptive scaling. AdamW fixes this → better generalization.
# RULE    : Prefer AdamW over Adam when using L2 regularization.
opt_adamw = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# ── optimizers.SGD(learning_rate, momentum, nesterov) ────────────────────────
# WHAT    : Stochastic Gradient Descent with optional momentum.
# PARAMS  :
#   momentum → accumulate past gradients; helps overcome local minima.
#   nesterov → Nesterov accelerated gradient; generally more accurate.
# PROS    : Often better generalization than Adam for CNNs with good LR schedule.
# CONS    : Needs careful LR tuning; slow without momentum.
opt_sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# ── optimizers.RMSprop(learning_rate, rho, epsilon) ──────────────────────────
# WHAT    : Divide gradient by root of exponential moving average of squares.
# PROS    : Works well for RNNs and non-stationary objectives.
# USE CASE: RNN training (historically); RL policy optimization.
opt_rms = optimizers.RMSprop(learning_rate=1e-3, rho=0.9, epsilon=1e-7)

# ── Gradient clipping (applied at optimizer level) ────────────────────────────
# WHAT    : Cap gradient norm / value before applying update.
# WHY     : Prevents exploding gradients in RNNs, transformers, deep nets.
# METHODS :
#   clipnorm  → clip by global gradient norm (preferred for RNNs)
#   clipvalue → clip each gradient individually by value
# RULE    : clipnorm=1.0 is a universal safe default.
opt_clipped = optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

# ── optimizers.Adagrad(learning_rate, initial_accumulator_value) ──────────────
# WHAT    : Accumulates squared gradients; rare features get larger updates.
# USE CASE: Sparse data, NLP with large vocabularies, embedding tables.
opt_adagrad = optimizers.Adagrad(learning_rate=0.01)

# ── optimizers.Adadelta(learning_rate, rho) ───────────────────────────────────
# WHAT    : Extension of Adagrad; adaptive without a global learning rate.
# USE CASE: When LR tuning is not feasible.
opt_adadelta = optimizers.Adadelta(learning_rate=1.0, rho=0.95)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 9 ── LEARNING RATE SCHEDULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Dynamically adjust the learning rate during training.
IMPORTANCE: A constant LR is almost never optimal. Decaying LR lets the model
             make large updates early and fine-tune later.

RULES:
  ✦ Warm-up prevents early training instability (large LR on bad init).
  ✦ Cosine annealing often reaches better minima than step decay.
  ✦ Cyclical LR can escape local minima.
  ✦ Pass schedule to optimizer's learning_rate argument.

REQUIREMENTS:
  • Schedule object must be an instance of
    tf.keras.optimizers.schedules.LearningRateSchedule or a callable.
"""

TOTAL_STEPS  = (55000 // BATCH_SIZE) * 20   # steps_per_epoch × epochs
WARMUP_STEPS = (55000 // BATCH_SIZE) * 2    # 2 warm-up epochs

# ── optimizers.schedules.CosineDecay(initial_lr, decay_steps, alpha) ──────────
# WHAT    : Decay LR following a cosine curve from initial_lr to alpha*initial_lr.
# WHY     : Smooth decay; often better than step decay; widely used in SOTA.
# PARAMS  :
#   initial_learning_rate → starting LR (after warm-up)
#   decay_steps           → number of steps to decay over
#   alpha                 → minimum LR as fraction of initial (default 0.0)
cosine_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=TOTAL_STEPS,
    alpha=0.0    # decays to 0
)

# ── optimizers.schedules.CosineDecayRestarts ──────────────────────────────────
# WHAT    : Cosine decay with periodic restarts (SGDR).
# WHY     : Each restart can escape local minima; often finds flatter minima.
# USE CASE: Long training runs, ensemble building via snapshots.
cosine_restarts = optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=TOTAL_STEPS // 4,
    t_mul=2.0,    # multiply restart period each cycle
    m_mul=0.9,    # decay peak LR each restart
)

# ── optimizers.schedules.ExponentialDecay ────────────────────────────────────
# WHAT    : Multiply LR by decay_rate every decay_steps.
# MATH    : lr = initial_lr * decay_rate^(step / decay_steps)
# RULE    : staircase=True → discrete steps; False → smooth continuous decay.
exp_decay = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# ── Custom LR schedule (warm-up + cosine) ────────────────────────────────────
class WarmupCosineSchedule(optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up for warmup_steps, then cosine decay.
    IMPORTANCE: Warm-up prevents large gradient steps when weights are random;
                 cosine decay ensures smooth final convergence.
    USE CASE  : Transformer training (GPT, BERT standard recipe).
    """
    def __init__(self, peak_lr, total_steps, warmup_steps):
        super().__init__()
        self.peak_lr      = peak_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step     = tf.cast(step, tf.float32)
        # Linear warm-up phase
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        # Cosine decay phase
        progress  = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = self.peak_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {"peak_lr": self.peak_lr,
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}

lr_schedule = WarmupCosineSchedule(
    peak_lr=1e-3, total_steps=TOTAL_STEPS, warmup_steps=WARMUP_STEPS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 10 ── REGULARIZATION TECHNIQUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Prevent overfitting — model memorizing training data but failing on new data.
IMPORTANCE: Overfitting is the primary failure mode when training data is limited.
             Without regularization, deep networks memorize noise.

RULES:
  ✦ Start without regularization; add if val_loss >> train_loss.
  ✦ L2 (weight decay) on Dense/Conv kernels is universally safe.
  ✦ Dropout: 0.2–0.3 after conv layers; 0.4–0.5 after fully connected.
  ✦ Do NOT apply Dropout after BatchNorm (conflicting normalization).
  ✦ Data augmentation is the most powerful regularizer for images.

REQUIREMENTS:
  • Regularizers are passed to layer arguments.
  • The regularization loss is automatically added to total loss during training.
"""

# ── regularizers.L2(l2) ──────────────────────────────────────────────────────
# WHAT    : Add λ * Σw² to the loss. Penalizes large weights.
# WHY     : Large weights → model is overfit to specific training patterns.
#            L2 → weight values are smoothed → better generalization.
# MATH    : loss_total = loss_task + λ * ||W||²
reg_l2  = regularizers.L2(l2=1e-4)

# ── regularizers.L1(l1) ──────────────────────────────────────────────────────
# WHAT    : Add λ * Σ|w| to the loss. Promotes SPARSE weights (many zeros).
# WHY     : Automatic feature selection — unimportant features get zero weight.
# USE CASE: High-dimensional input, feature selection, compressed models.
reg_l1  = regularizers.L1(l1=1e-5)

# ── regularizers.L1L2(l1, l2) ────────────────────────────────────────────────
# WHAT    : Elastic Net — combines L1 sparsity and L2 smoothing.
# USE CASE: Best of both worlds; handles correlated features better than L1.
reg_l1l2 = regularizers.L1L2(l1=1e-5, l2=1e-4)

# ── layers.Dropout(rate) ─────────────────────────────────────────────────────
# (Detailed explanation in Phase 4a above)

# ── layers.SpatialDropout2D(rate) ────────────────────────────────────────────
# WHAT    : Drop entire feature MAPS (channels) instead of individual units.
# WHY     : Spatially adjacent activations are highly correlated in CNNs;
#            dropping whole channels is more effective than random units.
# RULE    : Use instead of Dropout for Conv layers.
spatial_drop = layers.SpatialDropout2D(rate=0.2)

# ── layers.ActivityRegularization(l1, l2) ────────────────────────────────────
# WHAT    : Add penalty based on activation VALUES (not weights).
# WHY     : Encourages sparse activations (efficient representations).
# USE CASE: Autoencoders, sparse coding, interpretability.
act_reg = layers.ActivityRegularization(l1=1e-5, l2=1e-5)

# ── layers.GaussianNoise(stddev) ─────────────────────────────────────────────
# WHAT    : Add zero-mean Gaussian noise to inputs during training.
# WHY     : Acts as implicit data augmentation / Tikhonov regularization.
# RULE    : Only active during training; no effect during inference.
gauss_noise = layers.GaussianNoise(stddev=0.1)

# ── layers.GaussianDropout(rate) ─────────────────────────────────────────────
# WHAT    : Multiply by noise ~ N(1, rate/(1-rate)) during training.
# WHY     : Multiplicative noise is more natural for log-scale data.
gauss_drop = layers.GaussianDropout(rate=0.1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 11 ── NORMALIZATION LAYERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Normalize activations to stabilize training and allow deeper networks.
IMPORTANCE: Without normalization, internal covariate shift causes gradients to
             vanish or explode across layers; training becomes very slow or fails.

COMPARISON TABLE:
  ┌──────────────────┬────────────────────────┬───────────────────────────────┐
  │ Layer            │ Normalizes over        │ Best use case                 │
  ├──────────────────┼────────────────────────┼───────────────────────────────┤
  │ BatchNorm        │ batch × spatial        │ CNNs, large batch sizes       │
  │ LayerNorm        │ feature dimension      │ Transformers, RNNs, NLP       │
  │ InstanceNorm     │ spatial per sample     │ Style transfer, GANs          │
  │ GroupNorm        │ group of channels      │ Small batches, detection      │
  └──────────────────┴────────────────────────┴───────────────────────────────┘

RULES:
  ✦ Batch normalization requires batch_size ≥ 16; breaks at batch_size=1.
  ✦ Layer normalization works at any batch size → preferred for inference-heavy.
  ✦ Place normalization BEFORE activation (pre-activation) in residual nets.
  ✦ Do not use BatchNorm and Dropout together (they conflict).
"""

# ── layers.BatchNormalization(axis, momentum, epsilon, center, scale) ──────────
# (See Phase 4a for full explanation)
# KEY EXTRA PARAMS:
#   center → if True, add learnable beta (shift) — always True
#   scale  → if True, add learnable gamma (scale) — set False after ReLU
#   trainable → set False to freeze BN stats for transfer learning
bn = layers.BatchNormalization(
    axis=-1,         # channel axis (channels-last)
    momentum=0.99,   # higher = slower stat update (stable for small datasets)
    epsilon=1e-3,    # numerical stability; larger if inputs have large variance
    center=True,     # learnable beta (bias)
    scale=True       # learnable gamma (scale)
)

# ── layers.LayerNormalization(axis, epsilon) ──────────────────────────────────
# WHAT    : Normalize across the feature dimension for each sample independently.
# WHY     : Batch-independent → works at batch_size=1; essential for RNNs/Transformers.
# RULE    : Default axis=-1 (last axis = features). For sequence data, axis=-1 = features.
ln = layers.LayerNormalization(axis=-1, epsilon=1e-6)

# ── layers.GroupNormalization(groups, axis) ───────────────────────────────────
# WHAT    : Split channels into groups; normalize within each group.
# WHY     : Works well at small batch sizes where BatchNorm fails.
# RULE    : groups must divide channel count evenly. groups=1 = LayerNorm; groups=C = InstanceNorm.
gn = layers.GroupNormalization(groups=32, axis=-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 12 ── MODEL COMPILATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Configure the training procedure (optimizer, loss, metrics).
IMPORTANCE: Compilation links the model to its training objective.
             Misconfigured compilation is a silent bug (wrong loss = wrong training).

RULES:
  ✦ from_logits=True in loss → more numerically stable (preferred).
  ✦ Match loss to output activation:
      softmax output  → from_logits=False
      no activation   → from_logits=True
  ✦ Multiple metrics can be monitored simultaneously.
  ✦ For multi-output models, pass dict of losses and metrics.
  ✦ run_eagerly=True for debugging (disables graph compilation — very slow).

REQUIREMENTS:
  • optimizer, loss are required arguments.
  • metrics is optional but critical for monitoring training.
"""

model = sequential_model   # use the sequential model for compilation demo

# ── model.compile(optimizer, loss, metrics, loss_weights, run_eagerly) ─────────
# WHAT    : Configures the model for training.
# PARAMS  :
#   optimizer    → optimizer instance or string name ('adam', 'sgd')
#   loss         → loss instance, string, or callable
#   metrics      → list of metrics to track during training and evaluation
#   loss_weights → dict/list of floats to weight losses (multi-output models)
#   run_eagerly  → True = disable @tf.function (for step-by-step debugging)
model.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4,
                                clipnorm=1.0),
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[
        # ── metrics.SparseCategoricalAccuracy() ───────────────────────────
        # WHAT    : Fraction of correct top-1 predictions (integer labels).
        # WHY     : Most interpretable classification metric.
        metrics.SparseCategoricalAccuracy(name="accuracy"),

        # ── metrics.SparseTopKCategoricalAccuracy(k) ──────────────────────
        # WHAT    : Fraction where true class is in top-k predictions.
        # USE CASE: ImageNet evaluation (top-1 and top-5 accuracy).
        metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
    ],
    run_eagerly=False   # True for debugging; False for speed
)

print("\nModel compiled successfully.")
model.summary(expand_nested=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 13 ── CALLBACKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Automate actions at specific points during training (epoch start/end,
             batch start/end, training start/end).
IMPORTANCE: Without callbacks, you must manually monitor training and intervene.
             Callbacks automate early stopping, checkpointing, and LR scheduling.

RULES:
  ✦ ModelCheckpoint with save_best_only=True saves only improved models.
  ✦ EarlyStopping restore_best_weights=True reverts to best checkpoint.
  ✦ Order of callbacks in list matters for some interactions.
  ✦ TensorBoard log_dir should be unique per run (include timestamp).

REQUIREMENTS:
  • Pass as list to model.fit(callbacks=[...]).
  • Callbacks receive training state via logs dict.
"""

LOG_DIR  = f"/tmp/tb_logs/{int(time.time())}"
CKPT_DIR = "/tmp/best_model.keras"

training_callbacks = [

    # ── callbacks.ModelCheckpoint(filepath, monitor, save_best_only) ──────────
    # WHAT    : Save model weights (or full model) at specified intervals.
    # PARAMS  :
    #   filepath       → path with optional format keys {epoch}, {val_loss:.4f}
    #   monitor        → metric name to watch for improvement
    #   save_best_only → only save when monitored metric improves
    #   save_weights_only → save only weights (smaller file) or full model
    #   mode           → 'min' (for loss) | 'max' (for accuracy) | 'auto'
    # RULE    : Use .keras extension for full Keras format (recommended).
    callbacks.ModelCheckpoint(
        filepath=CKPT_DIR,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1
    ),

    # ── callbacks.EarlyStopping(monitor, patience, restore_best_weights) ──────
    # WHAT    : Stop training when monitored metric stops improving.
    # PARAMS  :
    #   patience             → number of epochs to wait after last improvement
    #   min_delta            → minimum change to qualify as improvement
    #   restore_best_weights → revert to best epoch weights when stopping
    # IMPORTANCE: Prevents overfitting; saves compute by stopping early.
    # RULE    : patience should be > ReduceLROnPlateau patience.
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1
    ),

    # ── callbacks.ReduceLROnPlateau(monitor, factor, patience) ───────────────
    # WHAT    : Reduce LR by factor when metric plateaus.
    # PARAMS  :
    #   factor   → new_lr = lr * factor (e.g. 0.5 halves LR)
    #   patience → epochs with no improvement before reducing
    #   min_lr   → lower bound on LR
    #   cooldown → epochs to wait before resuming normal monitoring after LR change
    # RULE    : factor should be in (0, 1); 0.1–0.5 typical.
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        cooldown=2,
        verbose=1
    ),

    # ── callbacks.TensorBoard(log_dir, histogram_freq) ────────────────────────
    # WHAT    : Log scalars, histograms, images, graphs to TensorBoard.
    # PARAMS  :
    #   log_dir        → directory to write event files
    #   histogram_freq → log weight histograms every N epochs (0 = disable)
    #   write_graph    → save model graph (can be large)
    #   profile_batch  → batch range to profile; 0 = disable
    # USAGE   : tensorboard --logdir /tmp/tb_logs
    callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=0     # set to "5,10" to profile batches 5–10
    ),

    # ── callbacks.CSVLogger(filename, append) ────────────────────────────────
    # WHAT    : Stream epoch results to a CSV file for offline analysis.
    callbacks.CSVLogger("/tmp/training_log.csv", append=False),

    # ── callbacks.TerminateOnNaN() ───────────────────────────────────────────
    # WHAT    : Stop training immediately if loss becomes NaN.
    # WHY     : NaN loss means training is irrecoverably broken (exploding grad).
    # RULE    : Always include — cheap insurance.
    callbacks.TerminateOnNaN(),
]

# ── Custom Callback ───────────────────────────────────────────────────────────
class LRMonitorCallback(callbacks.Callback):
    """
    Custom callback: print current learning rate after each epoch.
    USE CASE: Debug LR schedules; confirm LR decay is working.
    HOOKS AVAILABLE:
      on_train_begin / on_train_end
      on_epoch_begin(epoch, logs) / on_epoch_end(epoch, logs)
      on_batch_begin(batch, logs) / on_batch_end(batch, logs)
      on_test_begin / on_test_end / on_predict_begin / on_predict_end
    """
    def on_epoch_end(self, epoch, logs=None):
        # self.model → access to the model being trained
        # optimizer.learning_rate → may be a schedule or scalar
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "__call__"):
            # It's a schedule — call it with current step
            current = lr(self.model.optimizer.iterations).numpy()
        else:
            current = float(lr.numpy() if hasattr(lr, 'numpy') else lr)
        print(f"\n  [LRMonitor] Epoch {epoch+1} LR = {current:.2e}  "
              f"| val_acc = {logs.get('val_accuracy', 0):.4f}")

training_callbacks.append(LRMonitorCallback())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 14 ── TRAINING: model.fit()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Run the gradient descent loop to optimize model weights.
IMPORTANCE: This is where learning happens. Forward pass → compute loss →
             backpropagate gradients → update weights → repeat.

WHAT HAPPENS EACH BATCH:
  1. Forward pass   : input → layers → predictions (logits/probabilities)
  2. Loss compute   : loss_fn(y_true, y_pred) → scalar loss value
  3. Backward pass  : tf.GradientTape records ops; tape.gradient computes ∂loss/∂W
  4. Gradient clip  : (optional) clip to prevent explosion
  5. Weight update  : optimizer.apply_gradients → W = W - lr * grad

RULES:
  ✦ Shuffle training data every epoch (handled by tf.data or shuffle=True).
  ✦ Validate every epoch to detect overfitting early.
  ✦ An EPOCH = one full pass through ALL training samples.
  ✦ training=True during fit (enables Dropout and BN batch stats).
  ✦ training=False during evaluate/predict (uses running stats, no Dropout).

REQUIREMENTS:
  • model must be compiled before calling fit().
  • x and y must be compatible (same number of samples).
  • batch_size or steps_per_epoch must be specified.
"""

print("\n" + "="*60)
print("  PHASE 14 — TRAINING")
print("="*60)

# ── model.fit(x, y, batch_size, epochs, validation_data, callbacks) ────────────
# WHAT    : Train the model for a fixed number of epochs.
# PARAMS  :
#   x                → input data (numpy, tensor, tf.data.Dataset)
#   y                → target labels (None if using Dataset with labels)
#   batch_size       → samples per gradient update (ignored if using Dataset)
#   epochs           → number of complete passes through training data
#   validation_data  → (x_val, y_val) or Dataset for validation after each epoch
#   validation_split → float 0-1; use fraction of x as val (only with numpy)
#   callbacks        → list of Callback instances
#   class_weight     → dict {class: weight} for imbalanced datasets
#   sample_weight    → per-sample loss weighting (numpy array)
#   initial_epoch    → start epoch (useful for resuming training)
#   steps_per_epoch  → batches per epoch (used with infinite datasets)
#   verbose          → 0=silent, 1=progress bar, 2=one line per epoch
# RETURNS : History object (stores loss/metric values per epoch)
history = model.fit(
    train_ds,                     # tf.data.Dataset (includes labels)
    epochs=3,                     # small for demo; use 50–200 for real training
    validation_data=val_ds,
    callbacks=training_callbacks,
    verbose=1,
)

# ── history object ────────────────────────────────────────────────────────────
# WHAT    : Contains training and validation metrics for every epoch.
# USAGE   : Plot loss curves to diagnose underfitting / overfitting.
print(f"\nHistory keys: {list(history.history.keys())}")
print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final val accuracy  : {history.history['val_accuracy'][-1]:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 15 ── CUSTOM TRAINING LOOP (GradientTape)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Full manual control over the training process.
IMPORTANCE: Required when model.fit() is insufficient:
             • GANs (two alternating optimizers)
             • Meta-learning (MAML — gradient through gradient)
             • Reinforcement Learning (custom reward shaping)
             • Multi-task with complex loss weighting
             • Gradient accumulation (simulate large batches on small GPU)

RULES:
  ✦ Always wrap forward pass inside GradientTape context.
  ✦ Call model(x, training=True) — NOT model.predict().
  ✦ tape.gradient() returns None for variables not connected to loss.
  ✦ optimizer.apply_gradients() must receive (gradient, variable) pairs.
  ✦ Use @tf.function for speed (compiles loop body to graph).

REQUIREMENTS:
  • Must manually call metric.update_state() and metric.reset_state().
  • Must manually log metrics (history, TensorBoard, print).
"""

# Setup a fresh model and optimizer for custom loop demo
custom_model = build_functional_model()
custom_opt   = optimizers.AdamW(1e-3, weight_decay=1e-4, clipnorm=1.0)
custom_loss  = losses.SparseCategoricalCrossentropy(from_logits=False)

# ── Metric objects for manual tracking ────────────────────────────────────────
# metrics.Mean() → tracks running average of scalar values
# metrics.SparseCategoricalAccuracy() → tracks accuracy with integer labels
train_loss_metric = metrics.Mean(name="train_loss")
train_acc_metric  = metrics.SparseCategoricalAccuracy(name="train_acc")
val_loss_metric   = metrics.Mean(name="val_loss")
val_acc_metric    = metrics.SparseCategoricalAccuracy(name="val_acc")

@tf.function   # ← compile to graph for speed; remove for step-by-step debugging
def custom_train_step(x_batch, y_batch):
    """
    Single forward + backward + update step.
    @tf.function: compiled to TF graph → ~2-10× faster than eager.
    RULE: Cannot use Python control flow that depends on tensor VALUES inside
           @tf.function (use tf.cond, tf.while_loop instead).
    """
    # ── tf.GradientTape() ─────────────────────────────────────────────────────
    # WHAT    : Context manager that records all differentiable operations.
    # HOW     : TF builds a computation graph of ops for automatic differentiation.
    # PARAMS  :
    #   persistent → True allows multiple tape.gradient() calls (costs more memory).
    #                False (default) → tape destroyed after first gradient call.
    #   watch_accessed_variables → True (default) auto-watches all tf.Variables.
    with tf.GradientTape() as tape:
        # Forward pass — training=True enables Dropout and BN batch stats
        predictions = custom_model(x_batch, training=True)

        # Compute loss (scalar)
        step_loss = custom_loss(y_batch, predictions)

        # Add regularization losses (L1/L2 from kernel_regularizer)
        # model.losses → list of regularization losses registered by layers
        step_loss += tf.add_n(custom_model.losses) if custom_model.losses else 0.0

    # ── tape.gradient(target, sources) ───────────────────────────────────────
    # WHAT    : Compute ∂target/∂sources using reverse-mode auto-differentiation.
    # RETURNS : List of gradient tensors (same structure as sources).
    # RULE    : Gradients are None for sources not connected to target in graph.
    gradients = tape.gradient(step_loss, custom_model.trainable_variables)

    # ── Filter None gradients ─────────────────────────────────────────────────
    # RULE    : zip() with None gradients will crash apply_gradients.
    grads_and_vars = [(g, v) for g, v in
                      zip(gradients, custom_model.trainable_variables)
                      if g is not None]

    # ── optimizer.apply_gradients(grads_and_vars) ─────────────────────────────
    # WHAT    : Apply computed gradients to update weights.
    # RULE    : This must be called with (gradient, variable) pairs.
    custom_opt.apply_gradients(grads_and_vars)

    # Update metrics
    train_loss_metric.update_state(step_loss)
    train_acc_metric.update_state(y_batch, predictions)

@tf.function
def custom_val_step(x_batch, y_batch):
    """Validation step — no GradientTape, no weight updates."""
    predictions = custom_model(x_batch, training=False)  # training=False !
    step_loss   = custom_loss(y_batch, predictions)
    val_loss_metric.update_state(step_loss)
    val_acc_metric.update_state(y_batch, predictions)

print("\n" + "="*60)
print("  PHASE 15 — CUSTOM TRAINING LOOP")
print("="*60)

CUSTOM_EPOCHS = 2
for epoch in range(1, CUSTOM_EPOCHS + 1):
    t0 = time.time()

    # ── Reset metrics at start of each epoch ──────────────────────────────────
    # metric.reset_state() → clears accumulated values for new epoch.
    # RULE    : MUST reset before each epoch; forgetting accumulates wrong values.
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()
    val_loss_metric.reset_state()
    val_acc_metric.reset_state()

    for x_b, y_b in train_ds:
        custom_train_step(x_b, y_b)

    for x_b, y_b in val_ds:
        custom_val_step(x_b, y_b)

    # ── metric.result() → retrieve current metric value as scalar tensor ───────
    print(f"Epoch {epoch}/{CUSTOM_EPOCHS}  "
          f"loss={train_loss_metric.result():.4f}  "
          f"acc={train_acc_metric.result():.4f}  "
          f"val_loss={val_loss_metric.result():.4f}  "
          f"val_acc={val_acc_metric.result():.4f}  "
          f"({time.time()-t0:.1f}s)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 16 ── EVALUATION & METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Objectively assess model performance on unseen data.
IMPORTANCE: Training accuracy is meaningless — only test performance matters.
             Use appropriate metrics for the task; accuracy alone is misleading
             for imbalanced datasets.

RULES:
  ✦ NEVER evaluate on training data — it measures memorization, not learning.
  ✦ Use the test set only ONCE at the end of all training.
     Using it repeatedly introduces test set overfitting (selection bias).
  ✦ For imbalanced classes: use AUC, F1, Precision/Recall instead of accuracy.
  ✦ Report confidence intervals where possible (multiple random seeds).

REQUIREMENTS:
  • model.evaluate() requires model to be compiled.
  • Evaluation uses training=False (BN running stats, no Dropout).
"""

print("\n" + "="*60)
print("  PHASE 16 — EVALUATION")
print("="*60)

# ── model.evaluate(x, y, batch_size, return_dict) ─────────────────────────────
# WHAT    : Compute loss and metrics on provided data.
# PARAMS  :
#   x           → input data or tf.data.Dataset
#   verbose     → 0=silent, 1=progress bar
#   return_dict → True returns dict; False returns list [loss, metric1, metric2]
# RULE    : Pass test_ds, not train_ds.
results = model.evaluate(test_ds, verbose=1, return_dict=True)
print(f"\nTest loss    : {results['loss']:.4f}")
print(f"Test accuracy: {results['accuracy']:.4f}")

# ── Manual metric computation ─────────────────────────────────────────────────
# USE CASE: When you need metrics not available in Keras, or per-class metrics.

# metrics.AUC(curve, num_thresholds) ──────────────────────────────────────────
# WHAT    : Area Under the ROC or PR Curve.
# WHY     : AUC = probability model ranks positive above negative.
#            Threshold-independent; works for imbalanced classes.
# PARAMS  :
#   curve          → 'ROC' (TPR vs FPR) | 'PR' (Precision vs Recall)
#   num_thresholds → number of threshold points (higher = more accurate)
auc_metric = metrics.AUC(curve="ROC", num_thresholds=200, name="auc")

# metrics.Precision(thresholds) ───────────────────────────────────────────────
# WHAT    : TP / (TP + FP) — of predicted positives, fraction truly positive.
# WHY     : Important when false positives are costly (spam filter, fraud).
prec_metric = metrics.Precision(name="precision")

# metrics.Recall(thresholds) ──────────────────────────────────────────────────
# WHAT    : TP / (TP + FN) — of true positives, fraction correctly predicted.
# WHY     : Important when false negatives are costly (disease diagnosis).
rec_metric = metrics.Recall(name="recall")

# metrics.F1Score(num_classes, average) ───────────────────────────────────────
# WHAT    : Harmonic mean of Precision and Recall.
# WHY     : Balances precision and recall; best single metric for imbalanced tasks.
# PARAMS  : average → 'micro' | 'macro' | 'weighted'
f1_metric = metrics.F1Score(num_classes=10, average="macro", name="f1")

# metrics.MeanIoU(num_classes) ────────────────────────────────────────────────
# WHAT    : Mean Intersection-over-Union across all classes.
# USE CASE: Semantic segmentation evaluation.
miou = metrics.MeanIoU(num_classes=10)

# metrics.MeanAbsoluteError() / metrics.RootMeanSquaredError() ─────────────────
# USE CASE: Regression tasks.
mae_m  = metrics.MeanAbsoluteError(name="mae")
rmse_m = metrics.RootMeanSquaredError(name="rmse")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 17 ── PREDICTION & INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Use the trained model to make predictions on new data.
IMPORTANCE: The final goal of training — deploy model to make useful predictions.

RULES:
  ✦ Always use model(x, training=False) or model.predict() for inference.
     training=True would activate Dropout and use batch BN stats → wrong results.
  ✦ Normalize/preprocess inference data identically to training data.
  ✦ model.predict() is more memory-efficient for large datasets (streams batches).
  ✦ For real-time inference, prefer model(x) directly (avoids overhead).
  ✦ Apply tf.argmax to get class index from softmax outputs.
  ✦ Apply tf.sigmoid for binary classification; threshold at 0.5.

REQUIREMENTS:
  • Input shape must match training shape exactly.
  • No gradients tracked (no GradientTape context needed).
"""

print("\n" + "="*60)
print("  PHASE 17 — PREDICTION")
print("="*60)

# ── model.predict(x, batch_size, verbose) ─────────────────────────────────────
# WHAT    : Generate output predictions for all samples in x.
# PARAMS  :
#   x          → input array or Dataset
#   batch_size → internal batching for memory efficiency
#   verbose    → progress verbosity
# RETURNS : numpy array of predictions (not tensors).
# RULE    : Use for large datasets; streaming is handled automatically.
predictions_prob = model.predict(test_ds, verbose=0)   # shape [N, 10]

# ── tf.argmax(input, axis) ───────────────────────────────────────────────────
# WHAT    : Index of the maximum value along an axis.
# WHY     : Convert softmax probabilities to class indices.
predicted_classes = tf.argmax(predictions_prob, axis=1)   # [N]

# ── Direct model call (for single sample / real-time) ─────────────────────────
# WHAT    : Call model like a function.
# RULE    : Always pass training=False for inference.
# RETURNS : Tensor (not numpy); call .numpy() to convert.
sample = x_test[:1]                                  # shape [1, 28, 28, 1]
single_pred = model(sample, training=False)          # shape [1, 10]
confidence  = tf.reduce_max(single_pred, axis=-1)
pred_class  = tf.argmax(single_pred, axis=-1)

print(f"Sample predicted class: {pred_class.numpy()[0]}")
print(f"Sample confidence     : {confidence.numpy()[0]:.4f}")
print(f"True label            : {y_test.numpy()[0]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 18 ── DEBUGGING & VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Diagnose training problems, visualize learning progress.
IMPORTANCE: Training failure is common; systematic debugging is essential.
             Most bugs are silent (model trains but learns the wrong thing).

COMMON PROBLEMS & DIAGNOSIS:
  Loss = NaN            → exploding gradients; reduce LR; add gradient clipping.
  Loss not decreasing   → LR too low; wrong loss function; data issue.
  Train acc high,       → overfitting; add regularization; more data; less model.
    val acc low
  Train acc low,        → underfitting; bigger model; more epochs; better LR.
    val acc ≈ train acc
  Loss oscillates wildly→ LR too high; reduce by 10×.

RULES:
  ✦ Plot train vs val loss every experiment — it's the most informative diagnostic.
  ✦ Check gradient norms — should be O(1); if O(100+) → exploding.
  ✦ Visualize weight histograms to detect dead neurons (all zeros → dead).
  ✦ Use tf.debugging.check_numerics to catch NaN/Inf early.
"""

def plot_training_history(history, save_path="/tmp/training_curves.png"):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history.history["loss"],     label="Train Loss", lw=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",   lw=2)
    axes[0].set_title("Loss Curves", fontsize=14)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history.history["accuracy"],     label="Train Acc", lw=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc",   lw=2)
    axes[1].set_title("Accuracy Curves", fontsize=14)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to {save_path}")
    plt.close()

plot_training_history(history)

# ── Gradient norm monitoring (inside custom loop) ─────────────────────────────
def check_gradient_norms(model_to_check, sample_x, sample_y, loss_fn):
    """
    Compute and print gradient norms for each layer.
    IMPORTANCE: Detect vanishing gradients (norm → 0) or
                 exploding gradients (norm → ∞) in specific layers.
    """
    with tf.GradientTape() as tape:
        preds = model_to_check(sample_x, training=True)
        loss  = loss_fn(sample_y, preds)
    grads = tape.gradient(loss, model_to_check.trainable_variables)

    print("\n─── Gradient Norms ───")
    for var, grad in zip(model_to_check.trainable_variables, grads):
        if grad is not None:
            norm = tf.norm(grad).numpy()
            status = "✓" if 0.001 < norm < 100 else "⚠"
            print(f"  {status} {var.name:50s} norm={norm:.4f}")

x_sample = tf.cast(x_train_s[:32], tf.float32)
y_sample  = y_train_s[:32]
check_gradient_norms(
    model,
    x_sample,
    y_sample,
    losses.SparseCategoricalCrossentropy(from_logits=False)
)

# ── tf.debugging ─────────────────────────────────────────────────────────────
@tf.function
def safe_forward(x, model_ref):
    """
    tf.debugging functions become graph ops — run at graph speed.
    RULE    : Use in production if NaN is a risk (e.g., custom loss with log).
    """
    # tf.debugging.check_numerics(tensor, message) ────────────────────────────
    # WHAT    : Raise InvalidArgumentError if tensor contains NaN or Inf.
    x = tf.debugging.check_numerics(x, message="Input contains NaN/Inf!")

    out = model_ref(x, training=False)

    # tf.debugging.assert_greater(x, y, message) ─────────────────────────────
    # WHAT    : Assert x > y element-wise. Raises on failure.
    # USE CASE: Validate probabilities are positive before taking log.
    tf.debugging.assert_greater(
        out, tf.zeros_like(out), message="Probabilities must be > 0")

    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 19 ── MODEL SAVING & LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Persist trained model weights and architecture for later use.
IMPORTANCE: Training is expensive — save checkpoints to avoid re-training.
             Production deployment requires a serialized model.

FORMAT COMPARISON:
  ┌─────────────────────┬────────────────────────────────┬───────────────────┐
  │ Format              │ Saves                          │ Use when          │
  ├─────────────────────┼────────────────────────────────┼───────────────────┤
  │ .keras (native)     │ arch + weights + optimizer     │ Keras workflows   │
  │ SavedModel          │ arch + weights + graph + sigs  │ TF Serving, C++   │
  │ .weights.h5 (HDF5)  │ weights only                   │ Transfer learning │
  │ TFLite              │ optimized graph for mobile     │ Edge/mobile deploy│
  └─────────────────────┴────────────────────────────────┴───────────────────┘

RULES:
  ✦ Always save the BEST checkpoint (not the last epoch).
  ✦ .keras format is recommended for new projects (TF 2.12+).
  ✦ SavedModel for cross-language serving (C++, Java, Go).
  ✦ Include preprocessing in saved model for deployment consistency.
"""

# ── model.save(filepath) ─────────────────────────────────────────────────────
# WHAT    : Save full model (architecture + weights + training config).
# PARAMS  : filepath → .keras (native Keras v3) or directory (SavedModel)
# RULE    : .keras format is recommended; SavedModel for TF Serving.
model.save("/tmp/model_full.keras")                   # Keras v3 format
model.save("/tmp/model_savedmodel/")                  # SavedModel format

# ── keras.models.load_model(filepath) ────────────────────────────────────────
# WHAT    : Reconstruct full model from disk.
# PARAMS  : custom_objects → dict of {name: class} for custom layers/losses/metrics.
# RULE    : Must provide custom_objects for any user-defined layer/loss/metric.
loaded_model = keras.models.load_model(
    "/tmp/model_full.keras",
    custom_objects={
        "FocalLoss": FocalLoss,
        "SqueezeExcitation": SqueezeExcitation,
    }
)
print(f"\nLoaded model: {loaded_model.name}")

# ── model.save_weights(filepath) / model.load_weights(filepath) ───────────────
# WHAT    : Save/load ONLY weights (not architecture or optimizer state).
# WHY     : Lighter than full save; used for transfer learning.
# RULE    : Model architecture must be identical when loading weights.
model.save_weights("/tmp/model.weights.h5")
model.load_weights("/tmp/model.weights.h5")

# ── tf.saved_model.save / tf.saved_model.load ────────────────────────────────
# WHAT    : Low-level SavedModel API — for tf.Module models or multi-signature export.
# WHY     : Required for TF Serving, TFLite conversion, TF.js export.
tf.saved_model.save(model, "/tmp/tf_saved_model")
raw_loaded = tf.saved_model.load("/tmp/tf_saved_model")

print(f"SavedModel signatures: {list(raw_loaded.signatures.keys())}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 20 ── EXPORT & DEPLOYMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
PURPOSE   : Convert and optimize model for target deployment environment.
IMPORTANCE: A model that only runs in Python is not production-ready.
             Deployment requires optimization for size, speed, and runtime.

DEPLOYMENT TARGETS:
  TF Serving    → high-throughput REST/gRPC server (datacenter)
  TFLite        → mobile (Android/iOS) and microcontrollers
  TF.js         → browser and Node.js
  ONNX          → cross-framework (PyTorch, TensorFlow, ONNX Runtime)
  TF-TRT        → NVIDIA GPU inference optimization (TensorRT)

RULES:
  ✦ Always benchmark latency and accuracy after quantization.
  ✦ INT8 quantization requires a representative dataset for calibration.
  ✦ Quantization may reduce accuracy — acceptable trade-off for 4× speedup.
  ✦ Verify TFLite model output matches original model before deployment.
"""

# ── TFLite Conversion ─────────────────────────────────────────────────────────

# tf.lite.TFLiteConverter.from_keras_model(model) ─────────────────────────────
# WHAT    : Create a converter object from a Keras model.
# PARAMS  : model → compiled Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ── Float32 TFLite (baseline, no optimization) ────────────────────────────────
tflite_float32 = converter.convert()
with open("/tmp/model_float32.tflite", "wb") as f:
    f.write(tflite_float32)
print(f"\nFloat32 TFLite size: {len(tflite_float32)/1024:.1f} KB")

# ── Dynamic-range INT8 quantization ───────────────────────────────────────────
# WHAT    : Post-training quantization; weights quantized to int8.
# WHY     : ~4× model size reduction; ~2–3× inference speedup on CPU.
# PARAMS  : tf.lite.Optimize.DEFAULT → applies best available optimization.
converter2                  = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.optimizations    = [tf.lite.Optimize.DEFAULT]
tflite_int8_dynamic         = converter2.convert()
with open("/tmp/model_int8_dynamic.tflite", "wb") as f:
    f.write(tflite_int8_dynamic)
print(f"INT8 dynamic TFLite : {len(tflite_int8_dynamic)/1024:.1f} KB")

# ── Full INT8 quantization with representative dataset ────────────────────────
# WHAT    : Quantize both weights AND activations to INT8.
# WHY     : Enables deployment on microcontrollers (no float support).
# REQUIREMENT: representative_dataset_gen must yield input tensors
#               that cover the full range of expected inputs.
def representative_dataset_gen():
    """
    Generator that yields representative inputs for calibration.
    RULE    : Use ~100–500 diverse samples (not all zeros!).
               Must match the input shape and dtype of the model.
    """
    for x_batch, _ in test_ds.take(50):     # 50 batches × BATCH_SIZE samples
        yield [tf.cast(x_batch, tf.float32)]

converter3 = tf.lite.TFLiteConverter.from_keras_model(model)
converter3.optimizations               = [tf.lite.Optimize.DEFAULT]
converter3.representative_dataset      = representative_dataset_gen
converter3.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter3.inference_input_type       = tf.uint8   # or tf.int8
converter3.inference_output_type      = tf.uint8

try:
    tflite_full_int8 = converter3.convert()
    with open("/tmp/model_full_int8.tflite", "wb") as f:
        f.write(tflite_full_int8)
    print(f"Full INT8 TFLite    : {len(tflite_full_int8)/1024:.1f} KB")
except Exception as e:
    print(f"Full INT8 note: {e}")

# ── TFLite Inference ──────────────────────────────────────────────────────────
def run_tflite_inference(tflite_path, test_input):
    """
    Run inference with a TFLite model.
    USE CASE: Verify TFLite model outputs match original model;
               benchmarking on device.
    """
    # tf.lite.Interpreter(model_path) ─────────────────────────────────────────
    # WHAT    : Load TFLite model into runtime.
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()          # RULE: must call before using interpreter

    inp_details = interp.get_input_details()    # list of dicts: index, shape, dtype
    out_details = interp.get_output_details()   # list of dicts: index, shape, dtype

    interp.set_tensor(inp_details[0]["index"],
                      tf.cast(test_input, inp_details[0]["dtype"]).numpy())
    interp.invoke()                    # run inference
    return interp.get_tensor(out_details[0]["index"])

tflite_out = run_tflite_inference("/tmp/model_float32.tflite",
                                    x_test[:1])
print(f"\nTFLite prediction: class {np.argmax(tflite_out, axis=1)[0]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPLETE WORKFLOW SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     NEURAL NETWORK WORKFLOW AT A GLANCE                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  0  Environment     tf.config, tf.random.set_seed                            ║
║  │                                                                           ║
║  1  Raw Data        tf.keras.datasets, tfds, tf.io                           ║
║  │                                                                           ║
║  2  Preprocessing   tf.cast, tf.one_hot, tf.expand_dims, normalization       ║
║  │                                                                           ║
║  3  tf.data         from_tensor_slices → cache → shuffle → map → batch       ║
║  │                                                    → prefetch             ║
║  │                                                                           ║
║  4  Architecture    Sequential | Functional | Subclassing | Custom Layer     ║
║  │  Layers          Conv2D, Dense, LSTM, Transformer, Dropout, BN, Pool      ║
║  │                                                                           ║
║  5  Init            GlorotUniform (tanh) | HeNormal (ReLU) | Orthogonal(RNN) ║
║  │                                                                           ║
║  6  Activations     relu → hidden | softmax/sigmoid → output                 ║
║  │                                                                           ║
║  7  Loss            SparseCatCE | CatCE | BinaryCE | MSE | Huber | Focal     ║
║  │                                                                           ║
║  8  Optimizer       AdamW (default) | SGD+momentum | RMSprop (RNN)           ║
║  │                                                                           ║
║  9  LR Schedule     WarmupCosine | CosineDecay | ExponentialDecay            ║
║  │                                                                           ║
║ 10  Regularization  L1/L2 | Dropout | SpatialDropout | GaussianNoise         ║
║  │                                                                           ║
║ 11  Normalization   BatchNorm (CNN) | LayerNorm (Transformer) | GroupNorm    ║
║  │                                                                           ║
║ 12  Compile         model.compile(optimizer, loss, metrics)                  ║
║  │                                                                           ║
║ 13  Callbacks       ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,       ║
║  │                  TensorBoard, CSVLogger, TerminateOnNaN, Custom           ║
║  │                                                                           ║
║ 14  Training        model.fit(train_ds, epochs, validation_data, callbacks)  ║
║  │                                                                           ║
║ 15  Custom Loop     GradientTape → gradient → clip → apply_gradients         ║
║  │                                                                           ║
║ 16  Evaluation      model.evaluate | AUC | Precision | Recall | F1 | MIoU    ║
║  │                                                                           ║
║ 17  Prediction      model.predict | model(x, training=False) | tf.argmax     ║
║  │                                                                           ║
║ 18  Debugging       gradient norms | loss curves | tf.debugging | TensorBoard║
║  │                                                                           ║
║ 19  Saving          .keras | SavedModel | .weights.h5                        ║
║  │                                                                           ║
║ 20  Deployment      TFLite float32 | INT8 dynamic | Full INT8 | TF Serving   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

print("\n✅  Complete Neural Network Workflow executed successfully.")
print(f"   TensorBoard logs : {LOG_DIR}")
print(f"   Best checkpoint  : {CKPT_DIR}")
print(f"   TFLite model     : /tmp/model_float32.tflite")
print(f"   Training curves  : /tmp/training_curves.png")
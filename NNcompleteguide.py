"""
================================================================================
ADVANCED TENSORFLOW NEURAL NETWORKS - COMPREHENSIVE GUIDE
================================================================================
Topics Covered:
  1.  Environment Setup & GPU Configuration
  2.  Data Pipelines (tf.data API)
  3.  Custom Layers & Models
  4.  Convolutional Neural Networks (CNN)
  5.  Recurrent Neural Networks (RNN / LSTM / GRU)
  6.  Transformers & Attention Mechanisms
  7.  Autoencoders (Vanilla + Variational)
  8.  Generative Adversarial Networks (GAN)
  9.  Transfer Learning & Fine-Tuning
  10. Custom Training Loops & Gradient Tape
  11. Custom Loss Functions & Metrics
  12. Callbacks & TensorBoard Logging
  13. Mixed Precision Training
  14. Model Regularization (Dropout, BatchNorm, L1/L2)
  15. Hyperparameter Tuning with Keras Tuner
  16. Model Saving, Loading & TF-Lite Export
  17. Distributed Training Strategy
================================================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, time, numpy as np, matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input, regularizers, mixed_precision
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

print(f"TensorFlow  : {tf.__version__}")
print(f"Keras       : {keras.__version__}")
print(f"GPUs found  : {tf.config.list_physical_devices('GPU')}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. GPU CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
def configure_gpu():
    """Enable memory growth so TF doesn't grab all VRAM at once."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU configured: {len(gpus)} device(s)")

configure_gpu()


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA PIPELINE  (tf.data API)
# ══════════════════════════════════════════════════════════════════════════════
def build_image_pipeline(image_paths, labels, batch_size=32,
                          img_size=(224, 224), augment=False):
    """
    High-performance input pipeline:
      • parallel file reads  (AUTOTUNE)
      • optional augmentation
      • prefetching
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def load_and_preprocess(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment_fn(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_synthetic_dataset(n_samples=5000, n_features=20, n_classes=5,
                              batch_size=64):
    """In-memory synthetic dataset for quick experiments."""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    return ds.shuffle(2000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CUSTOM LAYERS & MODELS
# ══════════════════════════════════════════════════════════════════════════════
class ResidualBlock(layers.Layer):
    """Pre-activation residual block with optional projection shortcut."""

    def __init__(self, filters, kernel_size=3, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.stride      = stride

        self.bn1   = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(filters, kernel_size, stride,
                                   padding="same", use_bias=False)

        self.bn2   = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size,
                                   padding="same", use_bias=False)

        self.shortcut = layers.Conv2D(filters, 1, stride, use_bias=False) \
                        if stride != 1 else None

    def call(self, x, training=False):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(self.relu1(self.bn1(x, training=training)))
        out = self.conv2(self.relu2(self.bn2(out, training=training)))
        return out + residual

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "stride": self.stride})
        return cfg


class SEBlock(layers.Layer):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        c = input_shape[-1]
        self.gap   = layers.GlobalAveragePooling2D()
        self.fc1   = layers.Dense(max(c // self.ratio, 1), activation="relu")
        self.fc2   = layers.Dense(c, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, c))

    def call(self, x):
        scale = self.gap(x)
        scale = self.fc2(self.fc1(scale))
        scale = self.reshape(scale)
        return x * scale


# ── Functional-API example: mini-ResNet ──────────────────────────────────────
def build_mini_resnet(input_shape=(32, 32, 3), num_classes=10):
    inp = Input(shape=input_shape)
    x   = layers.Conv2D(64, 3, padding="same", use_bias=False)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.ReLU()(x)

    for filters, stride in [(64, 1), (128, 2), (256, 2)]:
        x = ResidualBlock(filters, stride=stride)(x)
        x = SEBlock()(x)

    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="MiniResNet")


# ── Subclassed model with multiple outputs ───────────────────────────────────
class MultiTaskModel(Model):
    """Shared backbone → classification + regression heads."""

    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.backbone   = keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
        ])
        self.cls_head   = layers.Dense(n_classes, activation="softmax")
        self.reg_head   = layers.Dense(1)            # regression output

    def call(self, x, training=False):
        features = self.backbone(x, training=training)
        return self.cls_head(features), self.reg_head(features)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONVOLUTIONAL NEURAL NETWORKS
# ══════════════════════════════════════════════════════════════════════════════
def build_cnn_classifier(input_shape=(28, 28, 1), num_classes=10):
    """Deep CNN with depthwise-separable blocks."""
    inp = Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)

    # Depthwise-separable blocks
    for filters in [64, 128, 256]:
        x = layers.SeparableConv2D(filters, 3, padding="same",
                                   activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="CNN_Classifier")


# ══════════════════════════════════════════════════════════════════════════════
# 5. RECURRENT NEURAL NETWORKS  (RNN / LSTM / GRU)
# ══════════════════════════════════════════════════════════════════════════════
def build_sequence_model(vocab_size=10000, embed_dim=128, max_len=200,
                          num_classes=5):
    """Stacked Bi-LSTM with attention for sequence classification."""
    inp  = Input(shape=(max_len,))
    x    = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x    = layers.SpatialDropout1D(0.2)(x)

    # Stacked Bi-LSTM
    x    = layers.Bidirectional(
               layers.LSTM(128, return_sequences=True, dropout=0.2,
                           recurrent_dropout=0.1))(x)
    x    = layers.Bidirectional(
               layers.GRU(64, return_sequences=True, dropout=0.2))(x)

    # Bahdanau-style attention
    score  = layers.Dense(1, activation="tanh")(x)          # (B, T, 1)
    weight = layers.Softmax(axis=1)(score)                  # (B, T, 1)
    x      = layers.Multiply()([x, weight])                 # weighted sum
    x      = tf.reduce_sum(x, axis=1)                       # (B, F)

    x    = layers.Dense(128, activation="relu")(x)
    x    = layers.Dropout(0.4)(x)
    out  = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="BiLSTM_Attention")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRANSFORMER & MULTI-HEAD SELF-ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
class TransformerBlock(layers.Layer):
    """Standard transformer encoder block."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn    = layers.MultiHeadAttention(num_heads, embed_dim // num_heads)
        self.ff      = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.ln1     = layers.LayerNormalization(epsilon=1e-6)
        self.ln2     = layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = layers.Dropout(dropout)
        self.drop2   = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x  = self.ln1(x + self.drop1(attn_out, training=training))
        ff_out = self.ff(x)
        return self.ln2(x + self.drop2(ff_out, training=training))


def positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, ...], tf.float32)   # (1, T, D)


def build_transformer_classifier(vocab_size=10000, max_len=200,
                                   embed_dim=128, num_heads=4,
                                   ff_dim=256, num_blocks=3, num_classes=5):
    inp  = Input(shape=(max_len,))
    x    = layers.Embedding(vocab_size, embed_dim)(inp)
    x   += positional_encoding(max_len, embed_dim)
    x    = layers.Dropout(0.1)(x)

    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    x    = layers.GlobalAveragePooling1D()(x)
    x    = layers.Dense(128, activation="relu")(x)
    x    = layers.Dropout(0.3)(x)
    out  = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="Transformer_Classifier")


# ══════════════════════════════════════════════════════════════════════════════
# 7. AUTOENCODERS  (Vanilla + Variational)
# ══════════════════════════════════════════════════════════════════════════════
def build_autoencoder(input_dim=784, latent_dim=32):
    """Vanilla autoencoder."""
    # Encoder
    enc_inp = Input(shape=(input_dim,))
    x = layers.Dense(512, activation="relu")(enc_inp)
    x = layers.Dense(256, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="relu")(x)
    encoder = Model(enc_inp, z, name="encoder")

    # Decoder
    dec_inp = Input(shape=(latent_dim,))
    x = layers.Dense(256, activation="relu")(dec_inp)
    x = layers.Dense(512, activation="relu")(x)
    reconstruction = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = Model(dec_inp, reconstruction, name="decoder")

    autoencoder = Model(enc_inp, decoder(z), name="autoencoder")
    return encoder, decoder, autoencoder


class Sampling(layers.Layer):
    """Re-parameterisation trick: z = μ + ε·σ."""
    def call(self, inputs):
        mean, log_var = inputs
        eps = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps


class VAE(Model):
    """Variational Autoencoder."""

    def __init__(self, input_dim=784, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

        # Encoder
        self.enc_dense1 = layers.Dense(512, activation="relu")
        self.enc_dense2 = layers.Dense(256, activation="relu")
        self.z_mean     = layers.Dense(latent_dim)
        self.z_log_var  = layers.Dense(latent_dim)
        self.sampling   = Sampling()

        # Decoder
        self.dec_dense1     = layers.Dense(256, activation="relu")
        self.dec_dense2     = layers.Dense(512, activation="relu")
        self.dec_out        = layers.Dense(input_dim, activation="sigmoid")

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker]

    def encode(self, x):
        h = self.enc_dense2(self.enc_dense1(x))
        return self.z_mean(h), self.z_log_var(h)

    def decode(self, z):
        return self.dec_out(self.dec_dense2(self.dec_dense1(z)))

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.sampling([mean, log_var])
        return self.decode(z)

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            mean, log_var = self.encode(x)
            z = self.sampling([mean, log_var])
            x_recon = self.decode(z)

            recon = tf.reduce_mean(
                keras.losses.binary_crossentropy(x, x_recon))
            kl    = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mean) - tf.exp(log_var))
            loss  = recon + kl

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


# ══════════════════════════════════════════════════════════════════════════════
# 8. GENERATIVE ADVERSARIAL NETWORK  (DCGAN)
# ══════════════════════════════════════════════════════════════════════════════
def build_dcgan_generator(latent_dim=100, channels=1):
    model = keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, 5, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(channels, 5, strides=2,
                               padding="same", activation="tanh"),
    ], name="DCGAN_Generator")
    return model


def build_dcgan_discriminator(image_shape=(28, 28, 1)):
    model = keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding="same",
                      input_shape=image_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 5, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Dense(1),                   # logit for BCEwithLogits
    ], name="DCGAN_Discriminator")
    return model


class DCGAN(Model):
    def __init__(self, generator, discriminator, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.generator     = generator
        self.discriminator = discriminator
        self.latent_dim    = latent_dim
        self.bce           = keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")

    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_opt = g_optimizer
        self.d_opt = d_optimizer

    def train_step(self, real_images):
        batch = tf.shape(real_images)[0]
        noise = tf.random.normal((batch, self.latent_dim))

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake  = self.generator(noise, training=True)

            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake, training=True)

            d_loss = (self.bce(tf.ones_like(real_logits), real_logits) +
                      self.bce(tf.zeros_like(fake_logits), fake_logits))
            g_loss =  self.bce(tf.ones_like(fake_logits), fake_logits)

        self.d_opt.apply_gradients(
            zip(d_tape.gradient(d_loss, self.discriminator.trainable_variables),
                self.discriminator.trainable_variables))
        self.g_opt.apply_gradients(
            zip(g_tape.gradient(g_loss, self.generator.trainable_variables),
                self.generator.trainable_variables))

        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        return {m.name: m.result() for m in self.metrics}


# ══════════════════════════════════════════════════════════════════════════════
# 9. TRANSFER LEARNING & FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
def build_transfer_model(base_name="EfficientNetV2S",
                          num_classes=10, input_shape=(224, 224, 3)):
    """
    Phase 1: freeze base, train head.
    Phase 2: unfreeze top layers, fine-tune end-to-end.
    """
    preprocess = {
        "EfficientNetV2S": keras.applications.efficientnet_v2.preprocess_input,
        "MobileNetV3Large": keras.applications.mobilenet_v3.preprocess_input,
        "ResNet50V2": keras.applications.resnet_v2.preprocess_input,
    }
    base_cls = {
        "EfficientNetV2S": keras.applications.EfficientNetV2S,
        "MobileNetV3Large": keras.applications.MobileNetV3Large,
        "ResNet50V2": keras.applications.ResNet50V2,
    }

    inp  = Input(shape=input_shape)
    x    = preprocess[base_name](inp)
    base = base_cls[base_name](include_top=False, weights="imagenet",
                                input_tensor=x)
    base.trainable = False                 # Phase 1: frozen

    x    = base.output
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(256, activation="relu")(x)
    x    = layers.Dropout(0.4)(x)
    out  = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out, name=f"TransferModel_{base_name}")

    def unfreeze_top(n_layers=30):
        """Unfreeze last n_layers for fine-tuning (Phase 2)."""
        base.trainable = True
        for layer in base.layers[:-n_layers]:
            layer.trainable = False
        print(f"Unfroze last {n_layers} layers of {base_name}")

    model.unfreeze_top = unfreeze_top   # attach helper
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 10. CUSTOM TRAINING LOOP  (GradientTape)
# ══════════════════════════════════════════════════════════════════════════════
def custom_training_loop(model, train_ds, val_ds, epochs=10,
                          learning_rate=1e-3):
    optimizer   = keras.optimizers.AdamW(learning_rate, weight_decay=1e-4)
    loss_fn     = keras.losses.SparseCategoricalCrossentropy()
    train_acc   = keras.metrics.SparseCategoricalAccuracy()
    val_acc     = keras.metrics.SparseCategoricalAccuracy()
    train_loss  = keras.metrics.Mean()
    val_loss    = keras.metrics.Mean()

    @tf.function                           # compile to graph for speed
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            # Support multi-output models
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = loss_fn(y, logits)
            loss += sum(model.losses)      # regularisation losses
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss.update_state(loss)
        train_acc.update_state(y, logits)

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        loss = loss_fn(y, logits)
        val_loss.update_state(loss)
        val_acc.update_state(y, logits)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss.reset_state(); train_acc.reset_state()
        val_loss.reset_state();   val_acc.reset_state()

        for x_batch, y_batch in train_ds:
            train_step(x_batch, y_batch)
        for x_batch, y_batch in val_ds:
            val_step(x_batch, y_batch)

        tl = train_loss.result().numpy()
        ta = train_acc.result().numpy()
        vl = val_loss.result().numpy()
        va = val_acc.result().numpy()

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        print(f"Epoch {epoch:03d}/{epochs}  "
              f"train_loss={tl:.4f}  train_acc={ta:.4f}  "
              f"val_loss={vl:.4f}  val_acc={va:.4f}  "
              f"({time.time()-t0:.1f}s)")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# 11. CUSTOM LOSS FUNCTIONS & METRICS
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(keras.losses.Loss):
    """Focal loss for class-imbalanced datasets."""
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce      = -y_true * tf.math.log(y_pred)
        weight  = self.alpha * y_true * (1 - y_pred) ** self.gamma
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


class DiceCoefficient(keras.metrics.Metric):
    """Dice coefficient for segmentation tasks."""
    def __init__(self, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight("intersection", initializer="zeros")
        self.union        = self.add_weight("union",        initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.union.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def result(self):
        return (2 * self.intersection + self.smooth) / \
               (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


def contrastive_loss(margin=1.0):
    """Siamese network contrastive loss."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        pos    = y_true * tf.square(y_pred)
        neg    = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(pos + neg)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 12. CALLBACKS & TENSORBOARD
# ══════════════════════════════════════════════════════════════════════════════
class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up then cosine annealing."""
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr      = base_lr
        self.total_steps  = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.base_lr * step / self.warmup_steps
        cosine = self.base_lr * 0.5 * (
            1 + tf.cos(np.pi * (step - self.warmup_steps) /
                       (self.total_steps - self.warmup_steps)))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {"base_lr": self.base_lr,
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}


def get_standard_callbacks(log_dir="logs", ckpt_path="best_model.keras"):
    return [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss",
                        verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1,
                    write_graph=True, profile_batch="5,10"),
    ]


class GradCAMCallback(keras.callbacks.Callback):
    """Save Grad-CAM visualisations at end of each epoch."""
    def __init__(self, val_data, layer_name, save_dir="gradcam"):
        super().__init__()
        self.val_data   = val_data
        self.layer_name = layer_name
        self.save_dir   = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def compute_gradcam(model, image, class_idx, layer_name):
        grad_model = Model(model.inputs,
                           [model.get_layer(layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(image[np.newaxis])
            loss = preds[:, class_idx]
        grads  = tape.gradient(loss, conv_out)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam    = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        cam    = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
        return cam.numpy()

    def on_epoch_end(self, epoch, logs=None):
        for images, labels in self.val_data.take(1):
            img = images[0].numpy()
            lbl = tf.argmax(labels[0]).numpy()
            cam = self.compute_gradcam(
                self.model, img, lbl, self.layer_name)
            np.save(f"{self.save_dir}/epoch_{epoch:03d}.npy", cam)


# ══════════════════════════════════════════════════════════════════════════════
# 13. MIXED PRECISION TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def enable_mixed_precision(policy="mixed_float16"):
    """Enable FP16 training for 2-3× speed-up on modern GPUs."""
    mixed_precision.set_global_policy(policy)
    print(f"Mixed precision enabled: {policy}")
    print(f"Compute dtype : {mixed_precision.global_policy().compute_dtype}")
    print(f"Variable dtype: {mixed_precision.global_policy().variable_dtype}")


def build_mixed_precision_model(input_shape=(224, 224, 3), num_classes=10):
    """When using FP16, the final Dense must cast back to float32."""
    inp = Input(shape=input_shape)
    x   = layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
    x   = layers.GlobalAveragePooling2D()(x)
    # Cast to float32 before softmax for numerical stability
    x   = layers.Dense(num_classes, dtype="float32")(x)
    out = layers.Activation("softmax", dtype="float32")(x)
    return Model(inp, out, name="MixedPrecisionModel")


# ══════════════════════════════════════════════════════════════════════════════
# 14. REGULARISATION  (Dropout, BatchNorm, L1/L2, Spectral Norm)
# ══════════════════════════════════════════════════════════════════════════════
def build_regularised_model(input_dim=128, num_classes=10):
    """Demonstrates multiple regularisation techniques."""
    inp = Input(shape=(input_dim,))
    x   = layers.Dense(512,
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4))(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.4)(x)

    x   = layers.Dense(256,
                        kernel_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5))(x)
    x   = layers.LayerNormalization()(x)         # alternative to BatchNorm
    x   = layers.Activation("gelu")(x)
    x   = layers.AlphaDropout(0.3)(x)            # SELU-compatible dropout

    x   = layers.Dense(128, activation="selu",
                        kernel_initializer="lecun_normal")(x)

    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="Regularised_Model")


# ══════════════════════════════════════════════════════════════════════════════
# 15. HYPERPARAMETER TUNING  (Keras Tuner)
# ══════════════════════════════════════════════════════════════════════════════
def hyperparameter_tuning_example():
    """
    Requires: pip install keras-tuner
    Demonstrates Bayesian optimisation for architecture search.
    """
    try:
        import keras_tuner as kt
    except ImportError:
        print("Install keras-tuner: pip install keras-tuner")
        return

    def build_model(hp):
        units  = hp.Int("units", 64, 512, step=64)
        layers_n = hp.Int("num_layers", 1, 4)
        lr     = hp.Float("lr", 1e-4, 1e-2, sampling="log")
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)

        model = keras.Sequential()
        model.add(Input(shape=(20,)))
        for _ in range(layers_n):
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(5, activation="softmax"))

        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=20,
        directory="kt_logs",
        project_name="demo_tuning"
    )
    print(tuner.search_space_summary())
    return tuner   # call tuner.search(train_ds, validation_data=val_ds)


# ══════════════════════════════════════════════════════════════════════════════
# 16. MODEL SAVING, LOADING & TFLITE EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def save_load_export(model, save_path="saved_model"):
    """SavedModel, H5, TF-Lite, and TF-Lite quantised."""

    # ── SavedModel format (recommended) ──────────────────────────────────────
    model.save(save_path)
    loaded = keras.models.load_model(save_path,
                                      custom_objects={"ResidualBlock": ResidualBlock,
                                                       "SEBlock": SEBlock})
    print(f"SavedModel loaded: {loaded.name}")

    # ── Keras native .keras ───────────────────────────────────────────────────
    model.save(f"{save_path}.keras")

    # ── TensorFlow Lite ───────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    # ── TFLite + INT8 dynamic-range quantisation ──────────────────────────────
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant = converter.convert()
    with open("model_quant.tflite", "wb") as f:
        f.write(tflite_quant)

    print(f"TFLite model size     : {len(tflite_model)/1024:.1f} KB")
    print(f"TFLite quantised size : {len(tflite_quant)/1024:.1f} KB")


def tflite_inference(tflite_path, sample_input):
    """Run inference on a saved TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_detail  = interpreter.get_input_details()[0]
    out_detail  = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp_detail["index"],
                            sample_input.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out_detail["index"])


# ══════════════════════════════════════════════════════════════════════════════
# 17. DISTRIBUTED TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def distributed_training_example(train_ds_factory, num_classes=10,
                                   input_dim=20, epochs=5):
    """
    MirroredStrategy replicates model on all GPUs and all-reduces gradients.
    For multi-machine training use MultiWorkerMirroredStrategy.
    """
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    # Create dataset OUTSIDE strategy scope; batch per-replica
    global_batch = 64 * strategy.num_replicas_in_sync
    dist_ds = strategy.experimental_distribute_dataset(
        train_ds_factory(batch_size=global_batch))

    model.fit(dist_ds, epochs=epochs)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# QUICK DEMO  (runs if executed directly)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RUNNING QUICK DEMO")
    print("="*60)

    # ── Synthetic data ────────────────────────────────────────────────────────
    N, D, C = 2000, 20, 5
    X_train = np.random.randn(N, D).astype(np.float32)
    y_train = np.random.randint(0, C, N)
    X_val   = np.random.randn(400, D).astype(np.float32)
    y_val   = np.random.randint(0, C, 400)

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE))
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(64).prefetch(tf.data.AUTOTUNE))

    # ── Build & compile a regularised model ───────────────────────────────────
    model = build_regularised_model(input_dim=D, num_classes=C)
    model.summary()

    lr_schedule = WarmUpCosineDecay(
        base_lr=1e-3,
        total_steps=len(train_ds) * 10,
        warmup_steps=len(train_ds) * 2
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(lr_schedule, weight_decay=1e-4),
        loss=FocalLoss(),
        metrics=["accuracy"],
    )

    # ── Keras .fit() with callbacks ───────────────────────────────────────────
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                         epochs=20, callbacks=callbacks, verbose=1)

    # ── Custom training loop demo ─────────────────────────────────────────────
    print("\n── Custom Training Loop ─────────────────────────────────")
    model2 = build_regularised_model(input_dim=D, num_classes=C)
    custom_training_loop(model2, train_ds, val_ds, epochs=5)

    # ── VAE demo ──────────────────────────────────────────────────────────────
    print("\n── VAE Training ─────────────────────────────────────────")
    vae = VAE(input_dim=D, latent_dim=8)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    vae.fit(train_ds, epochs=3, verbose=1)

    # ── Transformer ───────────────────────────────────────────────────────────
    print("\n── Transformer Classifier ───────────────────────────────")
    tf_model = build_transformer_classifier(
        vocab_size=1000, max_len=50, embed_dim=64,
        num_heads=4, ff_dim=128, num_blocks=2, num_classes=C)
    tf_model.summary()

    print("\n✅  All modules loaded and demo completed successfully.")
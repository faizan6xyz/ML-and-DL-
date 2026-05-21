import tensorflow as tf
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping
Dense = tf.keras.layers.Dense
Sequential = tf.keras.Sequential
load_model = tf.keras.models.load_model


model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
x= [[1]]
y= [1]
model.fit(x, y, epochs=10)

# Save model
model.save("nn_model.keras") 
'''
This saves:
    architecture
    weights
    optimizer
    metadata

.keras is the newer recommmed format than .h5
Keras developers prefer for:
    better compatibility
    future support
    improved serialization

So Keras uses its modern native .keras format.
'''

# save only the weights (Sometimes you only want the learned parameters.)
model.save_weights("nn_weights.weights.")
''' 
Weights are basically:
    large matrices
    tensor arrays

The HDF5 (.h5) format is very efficient for storing:
    numerical arrays
    matrices
    tensors

That is why Keras historically used .h5 for weights.
'''


# Automatically Save During Training
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_loss',
    save_best_only=True  # saves only the best model
)

# load model
model = load_model("nn_model.keras")

# load the weights
model.load_weights("nn_weights.weights.h5")

'''

Training
   ↓
Checkpoint saves (.ckpt)
   ↓
Best model found
   ↓
Save final model (.keras)
'''

'''
This will work when using the Sequential API or the Functional API, but unfortunately not when using Model subclassing. However, you can use save_weights() and load_weights() to at least save and restore the model parameters (but you will need to save and restore everything else yourself).
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

from create_dataset import make_train_dataset

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_groth(physical_devices[0], True)

model = Sequential(
    [
        Dense(units=16, input_shape=(1,), activation="relu"),
        Dense(units=32, activation="relu"),
        Dense(units=2, activation="softmax")
    ]
)

# model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
train_samples, train_labels = make_train_dataset()
model.fit(
    x=train_samples,
    y=train_labels, 
    batch_size=10,
    epochs=30,
    shuffle=True,
    verbose=2
)
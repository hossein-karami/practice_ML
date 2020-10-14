
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

from sklearn.metrics import confusion_matrix

from creating_dataset import (
    make_train_dataset, 
    make_validation_dataset,
    make_test_dataset
)
from creating_plots import plot_confusion_matrix

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_groth(physical_devices[0], True)


def save_model(model, name):
    model.save(f"./{name}.h5")

def load_keras_model(name):
    try:
        model = load_model(f"./{name}")
        return model

    except Exception:
        print(f"\n\n\n There is no keras model with this name. \n\n\n")


def make_keras_model():
    model = Sequential(
        [
            Dense(units=16, input_shape=(1,), activation="relu"),
            Dense(units=32, activation="relu"),
            Dense(units=2, activation="softmax")
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    train_samples, train_labels = make_train_dataset()
    val_data = make_validation_dataset()
    
    model.fit(
        x=train_samples,
        y=train_labels, 
        validation_data = val_data,
        batch_size=10,
        epochs=30,
        shuffle=True,
        verbose=2
    )

    return model


def do_inference(model, plotting=True):
    test_samples, test_labels = make_test_dataset()
    predictions = model.predict(x=test_samples, batch_size=10, verbose=0)
    pred_labels = np.argmax(predictions, axis=-1)
    conf_matrix = confusion_matrix(
        y_true = test_labels,
        y_pred = pred_labels
    )

    cm_plot_labels = ["Negative result", "Positive_results"]
    plot_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=cm_plot_labels,
        title="Confusion Matrix"
    )


def main():
    
    loading_model = False
    is_new_analysis = True
    saving_model = True
    model_name = "first_keras"

    if loading_model:
        model = load_keras_model(model_name)
    
    if is_new_analysis:
        model = make_keras_model()     
        do_inference(model, plotting=True)
    
    if saving_model:
        save_model(model, model_name)

if __name__ == "__main__":
    
    main()
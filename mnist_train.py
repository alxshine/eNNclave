import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import pandas as pd

import mnist_prepare_data
import utils

# hyperparameters
MODEL_FILE = 'models/mnist.h5'
HIST_FILE = 'hist_mnist.csv'
HIDDEN_NEURONS = 128
DROPOUT_RATIO = 0.4
NUM_EPOCHS = 3
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 2
BATCH_SIZE = 32

# dataset parameters
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

tf.compat.v1.set_random_seed(1337)

x_train, y_train = mnist_prepare_data.load_train_set()
x_test, y_test = mnist_prepare_data.load_test_set()

model = Sequential([
    layers.Input(INPUT_SHAPE),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss=sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=NUM_EPOCHS,
                    validation_data=(x_test, y_test),
                    validation_steps=VALIDATION_STEPS,
                    )

loss0, accuracy0 = model.evaluate(x_test, y_test)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(MODEL_FILE))
model.save(MODEL_FILE)

print("Saving history at: {}".format(HIST_FILE))
hist_df = pd.DataFrame(history.history)
with open(HIST_FILE, 'w+') as f:
    hist_df.to_csv(f)

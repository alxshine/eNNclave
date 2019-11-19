import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import tensorflow_datasets as tfds

import pandas as pd

tf.compat.v1.enable_eager_execution()

# hyperparameters
MODEL_FILE = 'models/mnist.h5'
HIST_FILE = 'hist_mnist.csv'
HIDDEN_NEURONS = 128
DROPOUT_RATIO = 0.4
NUM_EPOCHS = 100
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 2
BATCH_SIZE = 32

# dataset parameters
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

tf.compat.v1.set_random_seed(1337)

train_ds, test_ds = tfds.load('mnist',
                              split=[tfds.Split.TRAIN, tfds.Split.TEST],
                              as_supervised=True)
train_ds = train_ds.shuffle(buffer_size=2*BATCH_SIZE).repeat().batch(
    BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

model = Sequential([
    layers.Input(INPUT_SHAPE),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
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

history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_ds,
                    validation_steps=VALIDATION_STEPS,
                    )

loss0, accuracy0 = model.evaluate(test_ds)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(MODEL_FILE))
model.save(MODEL_FILE)

print("Saving history at: {}".format(HIST_FILE))
hist_df = pd.DataFrame(history.history)
with open(HIST_FILE, 'w+') as f:
    hist_df.to_csv(f)

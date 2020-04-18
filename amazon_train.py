from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np
import pandas as pd

import json
import os
from os.path import join
import plotille

from amazon_prepare_data import load_books
from amazon_eval import eval_true_accuracy

SEED = 1337
tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_FILE = 'models/amazon.h5'

NUM_WORDS = 20000
SEQUENCE_LENGTH = 500

DROPOUT_RATE = 0.3
EPOCHS = 14 # this is where we start to overfit

x_train, y_train, x_test, y_test = load_books(NUM_WORDS, SEQUENCE_LENGTH, seed = SEED)

model = Sequential()
model.add(layers.Embedding(NUM_WORDS, 32, input_length=SEQUENCE_LENGTH))
model.add(layers.SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.SeparableConv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())

model.add(layers.Dense(600, activation='relu'))
model.add(layers.Dropout(DROPOUT_RATE))
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dropout(DROPOUT_RATE))
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
print(model.summary())

hist = model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 2,
        validation_data = (x_test, y_test),
        validation_steps = 100,
        )

print(f"Saving model under {MODEL_FILE}")
model.save(MODEL_FILE)

history = hist.history
fig = plotille.Figure()
fig.width = 60
fig.height = 30
fig.set_x_limits(min_=0, max_=EPOCHS)

fig.plot(range(EPOCHS), history['mae'], label='Training MAE')
fig.plot(range(EPOCHS), history['val_mae'], label='Validation MAE')

# print(fig.show(legend=True))

#  _, mae, _ = model.evaluate(x_test, y_test, verbose = 0)
#  print(f"Final mean absolute error {mae}")

eval_true_accuracy(model, x_train, y_train, x_test, y_test)

from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np
import pandas as pd

import json
import os
from os.path import join
import plotille

from amazon_prepare_data import load_books, load_cds, rebuild_cds
from amazon_eval import eval_true_accuracy

SEED = 1337
tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_FILE = 'models/amazon.h5'

NUM_WORDS = 20000
SEQUENCE_LENGTH = 500

DROPOUT_RATE = 0.3
HIDDEN_NEURONS = 600
EPOCHS = 20 # this is where we start to overfit

x_train, y_train, x_test, y_test = load_cds(NUM_WORDS, SEQUENCE_LENGTH, seed = SEED)
x_train_orig, y_train_orig, x_test_orig, y_test_orig = load_books(NUM_WORDS, SEQUENCE_LENGTH, seed = SEED)

# get original accuracy
original_model = load_model(MODEL_FILE)
print("Original model on book data:")
eval_true_accuracy(original_model, x_train_orig, y_train_orig, x_test_orig, y_test_orig)
print("Original model on cd data:")
eval_true_accuracy(original_model, x_train, y_train, x_test, y_test)
print('\n')

# retrain last layer
print("\n\n####### Retraining last layer #######")
original_model = load_model(MODEL_FILE)

last_layer_model = Sequential()
for l in original_model.layers[:-1]:
    l.trainable = False
    last_layer_model.add(l)

for l in original_model.layers[-1:]:
    l.trainable = True
    last_layer_model.add(l)

last_layer_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
# print(last_layer_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = last_layer_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 0,
        )

print(f"Saving model under models/amazon_last_layer.h5")
last_layer_model.save('models/amazon_last_layer.h5')

eval_true_accuracy(last_layer_model, x_train, y_train, x_test, y_test)

# retrain dense layers
print("\n\n####### Retraining dense layers #######")
original_model = load_model(MODEL_FILE)

dense_model = Sequential()
for l in original_model.layers[:-6]:
    l.trainable = False
    dense_model.add(l)

for l in original_model.layers[-6:]:
    l.trainable = True
    dense_model.add(l)

dense_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
# print(dense_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = dense_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 0,
        )

print(f"Saving model under models/amazon_dense.h5")
dense_model.save('models/amazon_dense.h5')

eval_true_accuracy(dense_model, x_train, y_train, x_test, y_test)

# retrain conv and dense layers
print("\n\n####### Keeping only embedding and tokenizer #######")
original_model = load_model(MODEL_FILE)

conv_model = Sequential()
for l in original_model.layers[:1]:
    l.trainable = False
    conv_model.add(l)

for l in original_model.layers[1:]:
    l.trainable = True
    conv_model.add(l)

conv_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
# print(conv_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = conv_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 0,
        )

print(f"Saving model under models/amazon_conv.h5")
conv_model.save('models/amazon_conv.h5')


eval_true_accuracy(conv_model, x_train, y_train, x_test, y_test)

#  retrain entire network
print("\n\n####### Keeping only tokenizer #######")
original_model = load_model(MODEL_FILE)

full_model = Sequential()
for l in original_model.layers:
    l.trainable = True
    full_model.add(l)
full_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
# print(full_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = full_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 0,
        )

print(f"Saving model under models/amazon_full.h5")
full_model.save('models/amazon_full.h5')

eval_true_accuracy(full_model, x_train, y_train, x_test, y_test)

#  rebuild even tokenizer
print("\n\n####### rebuilding everything #######")
original_model = load_model(MODEL_FILE)
x_train, y_train, x_test, y_test, _ = rebuild_cds(NUM_WORDS, SEQUENCE_LENGTH, seed = SEED)

new_model = Sequential()
for l in original_model.layers:
    l.trainable = True
    new_model.add(l)
new_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
#  print(new_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = new_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 0,
        )

print(f"Saving model under models/amazon_new.h5")
new_model.save('models/amazon_new.h5')

eval_true_accuracy(new_model, x_train, y_train, x_test, y_test)

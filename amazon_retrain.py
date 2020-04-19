from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.optimizers import Adam, SGD
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
EPOCHS = 15 # this is where we start to overfit
RETRAIN_EPOCHS = 1

def train(model, x, y, epochs, retrain = False):
    if retrain:
        optimizer = SGD(lr=0.0001)
    else:
        optimizer = Adam()
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae', 'acc'])
    hist = model.fit(
            x,
            y,
            epochs = epochs,
            shuffle=True,
            verbose = 0,
            )
    return hist


x_train, y_train, x_test, y_test = load_cds(NUM_WORDS, SEQUENCE_LENGTH, seed = SEED)

# get original accuracy
original_model = load_model(MODEL_FILE)
print("Original model on cd data:")
eval_true_accuracy(original_model, x_train, y_train, x_test, y_test)

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

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"training for {EPOCHS} epochs")
train(last_layer_model, x_train, y_train, EPOCHS)
eval_true_accuracy(last_layer_model, x_train, y_train, x_test, y_test)
last_layer_model.save('models/amazon_last_layer.h5')

last_layer_ft_fixed = clone_model(last_layer_model)
print(f"retraining FIXED for {RETRAIN_EPOCHS} epochs")
train(last_layer_ft_fixed, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(last_layer_ft_fixed, x_train, y_train, x_test, y_test)
last_layer_ft_fixed.save('models/amazon_last_layer_ft_fixed.h5')

last_layer_ft_flexible = clone_model(last_layer_model)
last_layer_ft_flexible.trainable = True
print(f"retraining FLEXIBLE for {RETRAIN_EPOCHS} epochs")
train(last_layer_ft_flexible, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(last_layer_ft_flexible, x_train, y_train, x_test, y_test)
last_layer_ft_flexible.save('models/amazon_last_layer_ft_flexible.h5')

# retrain dense layers
print("\n\n\n####### Retraining dense layers #######")
original_model = load_model(MODEL_FILE)

dense_model = Sequential()
for l in original_model.layers[:-6]:
    l.trainable = False
    dense_model.add(l)

for l in original_model.layers[-6:]:
    l.trainable = True
    dense_model.add(l)

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"training for {EPOCHS} epochs")
train(dense_model, x_train, y_train, EPOCHS)
eval_true_accuracy(dense_model, x_train, y_train, x_test, y_test)
dense_model.save('models/amazon_dense.h5')

dense_ft_fixed = clone_model(dense_model)
print(f"retraining FIXED for {RETRAIN_EPOCHS} epochs")
train(dense_ft_fixed, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(dense_ft_fixed, x_train, y_train, x_test, y_test)
dense_ft_fixed.save('models/amazon_dense_ft_fixed.h5')

dense_ft_flexible = clone_model(dense_model)
dense_ft_flexible.trainable = True
print(f"retraining FLEXIBLE for {RETRAIN_EPOCHS} epochs")
train(dense_ft_flexible, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(dense_ft_flexible, x_train, y_train, x_test, y_test)
dense_ft_flexible.save('models/amazon_dense_ft_flexible.h5')

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

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"training for {EPOCHS} epochs")
train(conv_model, x_train, y_train, EPOCHS)
eval_true_accuracy(conv_model, x_train, y_train, x_test, y_test)
conv_model.save('models/amazon_conv.h5')

conv_ft_fixed = clone_model(conv_model)
print(f"retraining FIXED for {RETRAIN_EPOCHS} epochs")
train(conv_ft_fixed, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(conv_ft_fixed, x_train, y_train, x_test, y_test)
conv_ft_fixed.save('models/amazon_conv_ft_fixed.h5')

conv_ft_flexible = clone_model(conv_model)
conv_ft_flexible.trainable = True
print(f"retraining FLEXIBLE for {RETRAIN_EPOCHS} epochs")
train(conv_ft_flexible, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(conv_ft_flexible, x_train, y_train, x_test, y_test)
conv_ft_flexible.save('models/amazon_conv_ft_flexible.h5')

#  retrain entire network
print("\n\n####### Keeping only tokenizer #######")
original_model = load_model(MODEL_FILE)

full_model = Sequential()
for l in original_model.layers:
    l.trainable = True
    full_model.add(l)

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"training for {EPOCHS} epochs")
train(full_model, x_train, y_train, EPOCHS)
eval_true_accuracy(full_model, x_train, y_train, x_test, y_test)
full_model.save('models/amazon_full.h5')

full_ft_fixed = clone_model(full_model)
print(f"retraining FIXED for {RETRAIN_EPOCHS} epochs")
train(full_ft_fixed, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(full_ft_fixed, x_train, y_train, x_test, y_test)
full_ft_fixed.save('models/amazon_full_ft_fixed.h5')

full_ft_flexible = clone_model(full_model)
full_ft_flexible.trainable = True
print(f"retraining FLEXIBLE for {RETRAIN_EPOCHS} epochs")
train(full_ft_flexible, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(full_ft_flexible, x_train, y_train, x_test, y_test)
full_ft_flexible.save('models/amazon_full_ft_flexible.h5')

#  rebuild even tokenizer
print("\n\n####### rebuilding everything #######")
original_model = load_model(MODEL_FILE)
x_train, y_train, x_test, y_test, _ = rebuild_cds(NUM_WORDS, SEQUENCE_LENGTH, seed = SEED)

new_model = Sequential()
for l in original_model.layers:
    l.trainable = True
    new_model.add(l)

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"training for {EPOCHS} epochs")
train(new_model, x_train, y_train, EPOCHS)
eval_true_accuracy(new_model, x_train, y_train, x_test, y_test)
new_model.save('models/amazon_new.h5')

new_ft_fixed = clone_model(new_model)
print(f"retraining FIXED for {RETRAIN_EPOCHS} epochs")
train(new_ft_fixed, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(new_ft_fixed, x_train, y_train, x_test, y_test)
new_ft_fixed.save('models/amazon_new_ft_fixed.h5')

new_ft_flexible = clone_model(new_model)
new_ft_flexible.trainable = True
print(f"retraining FLEXIBLE for {RETRAIN_EPOCHS} epochs")
train(new_ft_flexible, x_train, y_train, RETRAIN_EPOCHS, retrain=True)
eval_true_accuracy(new_ft_flexible, x_train, y_train, x_test, y_test)
new_ft_flexible.save('models/amazon_new_ft_flexible.h5')

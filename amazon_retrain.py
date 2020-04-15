from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np
import pandas as pd

import json
import os
from os.path import join
import plotille

from amazon_prepare_data import load_books, load_cds
from amazon_eval import eval_true_accuracy

SEED = 1337
tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_FILE = 'models/amazon.h5'

NUM_WORDS = 10000
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
for i in range(len(original_model.layers[:-1])):
    l = original_model.layers[i]
    l.trainable = False
    last_layer_model.add(l)

last_layer_model.add(layers.Dense(1, activation='linear', name='final'))

last_layer_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
print(last_layer_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = last_layer_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 2,
        validation_data = (x_test, y_test),
        validation_steps = 100,
        )

print(f"Saving model under models/amazon_last_layer.h5")
last_layer_model.save('models/amazon_last_layer.h5')

history = hist.history
fig = plotille.Figure()
fig.width = 60
fig.height = 30
fig.set_x_limits(min_=0, max_=EPOCHS)

fig.plot(range(EPOCHS), history['mae'], label='Training MAE')
fig.plot(range(EPOCHS), history['val_mae'], label='Validation MAE')

print(fig.show(legend=True))

print("Generating true training accuracy")
train_predictions = last_layer_model.predict(x_train, verbose = 0)
train_cleaned_predictions = train_predictions.flatten().round()
train_acc = np.mean(train_cleaned_predictions == y_train)
train_mae = np.mean(np.abs(train_cleaned_predictions - y_train))

print(f'True training accuracy: {train_acc*100:.4}')
print(f'Training MAE: {train_mae:.4}')

print("Generating true test accuracy")
test_predictions = last_layer_model.predict(x_test, verbose = 0)
test_cleaned_predictions = test_predictions.flatten().round()
test_acc = np.mean(test_cleaned_predictions == y_test)
test_mae = np.mean(np.abs(test_cleaned_predictions - y_test))

print(f'True test accuracy: {test_acc*100:.4}')
print(f'Test MAE: {test_mae:.4}')

# retrain dense layers
print("\n\n####### Retraining dense layers #######")
original_model = load_model(MODEL_FILE)

dense_model = Sequential()
for i in range(len(original_model.layers[:-6])):
    l = original_model.layers[i]
    l.trainable = False
    dense_model.add(l)

dense_model.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
dense_model.add(layers.Dropout(DROPOUT_RATE))
dense_model.add(layers.Dense(150, activation='relu'))
dense_model.add(layers.Dropout(DROPOUT_RATE))
dense_model.add(layers.Dense(150, activation='relu'))
dense_model.add(layers.Dense(1, activation='linear'))

dense_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
print(dense_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = dense_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 2,
        validation_data = (x_test, y_test),
        validation_steps = 100,
        )

print(f"Saving model under models/amazon_dense.h5")
dense_model.save('models/amazon_dense.h5')

history = hist.history
fig = plotille.Figure()
fig.width = 60
fig.height = 30
fig.set_x_limits(min_=0, max_=EPOCHS)

fig.plot(range(EPOCHS), history['mae'], label='Training MAE')
fig.plot(range(EPOCHS), history['val_mae'], label='Validation MAE')

print(fig.show(legend=True))

print("Generating true training accuracy")
train_predictions = dense_model.predict(x_train, verbose = 0)
train_cleaned_predictions = train_predictions.flatten().round()
train_acc = np.mean(train_cleaned_predictions == y_train)
train_mae = np.mean(np.abs(train_cleaned_predictions - y_train))

print(f'True training accuracy: {train_acc*100:.4}')
print(f'Training MAE: {train_mae:.4}')

print("Generating true test accuracy")
test_predictions = dense_model.predict(x_test, verbose = 0)
test_cleaned_predictions = test_predictions.flatten().round()
test_acc = np.mean(test_cleaned_predictions == y_test)
test_mae = np.mean(np.abs(test_cleaned_predictions - y_test))

print(f'True test accuracy: {test_acc*100:.4}')
print(f'Test MAE: {test_mae:.4}')

# retrain conv and dense layers
print("\n\n####### Keeping only embedding and tokenizer #######")
original_model = load_model(MODEL_FILE)

conv_model = Sequential()
l = original_model.layers[0]
l.trainable = False
conv_model.add(l)

conv_model.add(layers.SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
conv_model.add(layers.MaxPooling1D(pool_size=2))
conv_model.add(layers.SeparableConv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
conv_model.add(layers.MaxPooling1D(pool_size=2))
conv_model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
conv_model.add(layers.MaxPooling1D(pool_size=2))
conv_model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
conv_model.add(layers.MaxPooling1D(pool_size=2))
conv_model.add(layers.GlobalAveragePooling1D())

conv_model.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
conv_model.add(layers.Dropout(DROPOUT_RATE))
conv_model.add(layers.Dense(150, activation='relu'))
conv_model.add(layers.Dropout(DROPOUT_RATE))
conv_model.add(layers.Dense(150, activation='relu'))
conv_model.add(layers.Dense(1, activation='linear'))

conv_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
print(conv_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = conv_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 2,
        validation_data = (x_test, y_test),
        validation_steps = 100,
        )

print(f"Saving model under models/amazon_new.h5")
conv_model.save('models/amazon_new.h5')

history = hist.history
fig = plotille.Figure()
fig.width = 60
fig.height = 30
fig.set_x_limits(min_=0, max_=EPOCHS)

fig.plot(range(EPOCHS), history['mae'], label='Training MAE')
fig.plot(range(EPOCHS), history['val_mae'], label='Validation MAE')

print(fig.show(legend=True))

print("Generating true training accuracy")
train_predictions = conv_model.predict(x_train, verbose = 0)
train_cleaned_predictions = train_predictions.flatten().round()
train_acc = np.mean(train_cleaned_predictions == y_train)
train_mae = np.mean(np.abs(train_cleaned_predictions - y_train))

print(f'True training accuracy: {train_acc*100:.4}')
print(f'Training MAE: {train_mae:.4}')

print("Generating true test accuracy")
test_predictions = conv_model.predict(x_test, verbose = 0)
test_cleaned_predictions = test_predictions.flatten().round()
test_acc = np.mean(test_cleaned_predictions == y_test)
test_mae = np.mean(np.abs(test_cleaned_predictions - y_test))

print(f'True test accuracy: {test_acc*100:.4}')
print(f'Test MAE: {test_mae:.4}')

# retrain entire network
print("\n\n####### Keeping only tokenizer #######")
original_model = load_model(MODEL_FILE)

full_model.add(layers.Embedding(NUM_WORDS, 64, input_length=SEQUENCE_LENGTH))
full_model.add(layers.SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
full_model.add(layers.MaxPooling1D(pool_size=2))
full_model.add(layers.SeparableConv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
full_model.add(layers.MaxPooling1D(pool_size=2))
full_model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
full_model.add(layers.MaxPooling1D(pool_size=2))
full_model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
full_model.add(layers.MaxPooling1D(pool_size=2))
full_model.add(layers.GlobalAveragePooling1D())

full_model.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
full_model.add(layers.Dropout(DROPOUT_RATE))
full_model.add(layers.Dense(150, activation='relu'))
full_model.add(layers.Dropout(DROPOUT_RATE))
full_model.add(layers.Dense(150, activation='relu'))
full_model.add(layers.Dense(1, activation='linear'))

full_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
print(full_model.summary())

tf.random.set_seed(SEED)
np.random.seed(SEED)

hist = full_model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        verbose = 2,
        validation_data = (x_test, y_test),
        validation_steps = 100,
        )

print(f"Saving model under models/amazon_new.h5")
full_model.save('models/amazon_new.h5')

history = hist.history
fig = plotille.Figure()
fig.width = 60
fig.height = 30
fig.set_x_limits(min_=0, max_=EPOCHS)

fig.plot(range(EPOCHS), history['mae'], label='Training MAE')
fig.plot(range(EPOCHS), history['val_mae'], label='Validation MAE')

print(fig.show(legend=True))

print("Generating true training accuracy")
train_predictions = full_model.predict(x_train, verbose = 0)
train_cleaned_predictions = train_predictions.flatten().round()
train_acc = np.mean(train_cleaned_predictions == y_train)
train_mae = np.mean(np.abs(train_cleaned_predictions - y_train))

print(f'True training accuracy: {train_acc*100:.4}')
print(f'Training MAE: {train_mae:.4}')

print("Generating true test accuracy")
test_predictions = full_model.predict(x_test, verbose = 0)
test_cleaned_predictions = test_predictions.flatten().round()
test_acc = np.mean(test_cleaned_predictions == y_test)
test_mae = np.mean(np.abs(test_cleaned_predictions - y_test))

print(f'True test accuracy: {test_acc*100:.4}')
print(f'Test MAE: {test_mae:.4}')

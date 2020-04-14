import tensorflow.keras.preprocessing.text as pre_text
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np
import pandas as pd

import json
import os
from os.path import join
import plotille

SEED = 1337
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATA_DIR = 'datasets/amazon'
MODEL_FILE = 'models/amazon.h5'

NUM_SAMPLES = 2000000
SEQUENCE_LENGTH = 500
NUM_WORDS = 10000
TOKENIZER_CONFIG_FILE = 'data/amazon_tokenizer_config.json'

DROPOUT_RATE = 0.3
HIDDEN_NEURONS = 600
EPOCHS = 10 # this is where we start to overfit
LARGE = False
TRAIN_SPLIT = 0.8

if LARGE:
    try:
        # load numpy matrices
        print(f"Trying to load large training data from {DATA_DIR}")

        x_train = np.load(join(DATA_DIR, 'x_train_large.npy'))
        y_train = np.load(join(DATA_DIR, 'y_train_large.npy'))
        x_test = np.load(join(DATA_DIR, 'x_test_large.npy'))
        y_test = np.load(join(DATA_DIR, 'y_test_large.npy'))
    except IOError:
        print("Not found, generating...")
        data = pd.read_pickle(os.path.join(DATA_DIR, 'large.pkl'))
        train_data = pd.DataFrame(columns=data.columns)
        test_data = pd.DataFrame(columns=data.columns)

        for i in range(1,5):
            current_data = data.where(data['rating'] == i)
            current_train_data = current_data.sample(frac=TRAIN_SPLIT, replace=False, random_state=SEED).dropna(how='all')
            current_test_data = current_data[~current_data.isin(current_train_data)].dropna(how='all')

            train_data = train_data.append(current_train_data)
            test_data = test_data.append(current_test_data)

        train_data = train_data.sample(frac=1, replace=False, random_state = SEED)
        test_data = test_data.sample(frac=1, replace=False, random_state = SEED)

        train_texts = train_data['text']
        y_train = np.array(train_data['rating'])

        test_texts = test_data['text']
        y_test = np.array(test_data['rating'])

        tokenizer = pre_text.Tokenizer(NUM_WORDS)
        tokenizer.fit_on_texts(train_texts)

        train_sequences = tokenizer.texts_to_sequences(train_texts)
        test_sequences = tokenizer.texts_to_sequences(test_texts)

        x_train = sequence.pad_sequences(train_sequences, maxlen=SEQUENCE_LENGTH)
        x_test = sequence.pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH)

        np.save(join(DATA_DIR, 'x_train_large.npy'), x_train)
        np.save(join(DATA_DIR, 'y_train_large.npy'), y_train)
        np.save(join(DATA_DIR, 'x_test_large.npy'), x_test)
        np.save(join(DATA_DIR, 'y_test_large.npy'), y_test)

else:
    try:
        # load numpy matrices
        print(f"Trying to load training data from {DATA_DIR}")

        x_train = np.load(join(DATA_DIR, 'x_train.npy'))
        y_train = np.load(join(DATA_DIR, 'y_train.npy'))
        x_test = np.load(join(DATA_DIR, 'x_test.npy'))
        y_test = np.load(join(DATA_DIR, 'y_test.npy'))
    except IOError:
        print("Not found, generating...")
        data = pd.read_pickle(os.path.join(DATA_DIR, 'reduced.pkl'))

        train_data = pd.DataFrame(columns=data.columns)
        test_data = pd.DataFrame(columns=data.columns)

        for i in range(1,5):
            current_data = data.where(data['rating'] == i)
            current_train_data = current_data.sample(frac=TRAIN_SPLIT, replace=False, random_state=SEED).dropna(how='all')
            current_test_data = current_data[~current_data.isin(current_train_data)].dropna(how='all')

            train_data = train_data.append(current_train_data)
            test_data = test_data.append(current_test_data)

        train_texts = train_data['text']
        y_train = np.array(train_data['rating'])

        test_texts = test_data['text']
        y_test = np.array(test_data['rating'])

        tokenizer = pre_text.Tokenizer(NUM_WORDS)
        tokenizer.fit_on_texts(train_texts)

        train_sequences = tokenizer.texts_to_sequences(train_texts)
        test_sequences = tokenizer.texts_to_sequences(test_texts)

        x_train = sequence.pad_sequences(train_sequences, maxlen=SEQUENCE_LENGTH)
        x_test = sequence.pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH)

        np.save(join(DATA_DIR, 'x_train.npy'), x_train)
        np.save(join(DATA_DIR, 'y_train.npy'), y_train)
        np.save(join(DATA_DIR, 'x_test.npy'), x_test)
        np.save(join(DATA_DIR, 'y_test.npy'), y_test)

print("DONE")

model = Sequential()
model.add(layers.Embedding(NUM_WORDS, 64, input_length=SEQUENCE_LENGTH))
model.add(layers.SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.SeparableConv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())

model.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
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

print(fig.show(legend=True))

#  _, mae, _ = model.evaluate(x_test, y_test, verbose = 0)
#  print(f"Final mean absolute error {mae}")

print("Generating true training accuracy")
train_predictions = model.predict(x_train, verbose = 0)
train_cleaned_predictions = train_predictions.flatten().round()
train_acc = np.mean(train_cleaned_predictions == y_train)
train_mae = np.mean(np.abs(train_cleaned_predictions - y_train))

print(f'True training accuracy: {train_acc*100:.4}')
print(f'Training MAE: {train_mae:.4}')

print("Generating true test accuracy")
test_predictions = model.predict(x_test, verbose = 0)
test_cleaned_predictions = test_predictions.flatten().round()
test_acc = np.mean(test_cleaned_predictions == y_test)
test_mae = np.mean(np.abs(test_cleaned_predictions - y_test))

print(f'True test accuracy: {test_acc*100:.4}')
print(f'Test MAE: {test_mae:.4}')

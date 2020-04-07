import tensorflow.keras.preprocessing.text as pre_text
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers

import numpy as np
import pandas as pd

import json
import os
import plotille

SEED = 1337

DATA_DIR = '/data/datasets/amazon'
PICKLE_FILE = 'reduced.pkl'

NUM_SAMPLES = 2000000
SEQUENCE_LENGTH = 500
NUM_WORDS = 10000
TOKENIZER_CONFIG_FILE = 'data/amazon_tokenizer_config.json'

DROPOUT_RATE = 0.3
HIDDEN_NEURONS = 300
EPOCHS = 300

data = pd.read_pickle(os.path.join(DATA_DIR, PICKLE_FILE))

split_index = int(0.8*len(data.index))
train_data = data[:split_index]['text']
y_train = np.array(data[:split_index]['rating'])
y_train -= 1 # move from [1,5] to [0,4]

test_data = data[split_index:]['text']
y_test = np.array(data[split_index:]['rating'])
y_test -= 1

tokenizer = pre_text.Tokenizer(NUM_WORDS)
tokenizer.fit_on_texts(train_data)

train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

x_train = sequence.pad_sequences(train_sequences, maxlen=SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH)

model = Sequential()
model.add(layers.Embedding(NUM_WORDS, 64, input_length=SEQUENCE_LENGTH))
model.add(layers.SeparableConv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())

model.add(layers.Dropout(DROPOUT_RATE))
model.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
model.add(layers.Dropout(DROPOUT_RATE))
model.add(layers.Dense(HIDDEN_NEURONS, activation='relu'))
model.add(layers.Dropout(DROPOUT_RATE))

model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

hist = model.fit(
        x_train,
        y_train,
        epochs = EPOCHS,
        shuffle=True,
        batch_size = 128,
        verbose = 1,
        validation_data = (x_train, y_train),
        validation_steps = 10,
        )

history = hist.history
fig = plotille.Figure()
fig.width = 60
fig.height = 30
fig.set_x_limits(min_=0, max_=EPOCHS)
fig.set_y_limits(min_=0, max_=1)

fig.plot(range(EPOCHS), history['acc'], label='Training accuracy')
fig.plot(range(EPOCHS), history['val_acc'], label='Validation accuracy')

print(fig.show(legend=True))

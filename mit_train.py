import tensorflow.keras.layers as layers
import tensorflow.keras.applications as apps
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf

import pandas as pd

tf.compat.v1.set_random_seed(1337)

from os.path import join
import os
import json

import utils

# global config
data_dir = 'data/mit'
label_file = join(data_dir, 'class_labels.json')
train_file = 'TrainImages.txt'
test_file = 'TestImages.txt'

# load label mapping
with open(label_file, 'r') as f:
    labels = json.load(f)


# load images
with open(join(data_dir, train_file), 'r') as f:
    train_images = [join(data_dir, l.strip()) for l in f]
with open(join(data_dir, test_file), 'r') as f:
    test_images = [join(data_dir, l.strip()) for l in f.readlines()]

# generate labels
train_labels = [labels[s.split('/')[-2]] for s in train_images]
test_labels = [labels[s.split('/')[-2]] for s in test_images]

# generate datasets
train_ds = utils.generate_dataset(train_images, train_labels)
test_ds = utils.generate_dataset(
    test_images, test_labels, shuffle=False, repeat=False)

# build model
MODEL_FILE = 'models/mit.h5'
HIST_FILE = 'hist_mit.csv'
HIDDEN_NEURONS = 2048
DROPOUT_RATIO=0.4
NUM_EPOCHS = 1000
STEPS_PER_EPOCH = 2

extractor = apps.VGG16(include_top=False, weights='imagenet',
                  input_shape=((224, 224, 3)))
extractor.trainable = False

dense = Sequential([
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(len(labels), activation='softmax')
])

model = Sequential([
    extractor,
    layers.GlobalAveragePooling2D(name='gap2d'),
    dense])

print('Hypeparameters:')
print('num_epochs: {}'.format(NUM_EPOCHS))
print('hidden_neurons: {}'.format(HIDDEN_NEURONS))
print('training set size: {}'.format(len(train_labels)))
print('test set size: {}'.format(len(test_labels)))
print()

model.compile(optimizer='adam',
              loss=sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_ds,
                    validation_steps=STEPS_PER_EPOCH)

loss0, accuracy0 = model.evaluate(test_ds)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(MODEL_FILE))
model.save(MODEL_FILE)

print("Saving history at: {}".format(HIST_FILE))
hist_df = pd.DataFrame(history.history)
with open(HIST_FILE, 'w+') as f:
    hist_df.to_csv(f)

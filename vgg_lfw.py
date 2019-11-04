# coding: utf-8

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow as tf
from tensorflow.keras.applications import VGG16
import numpy as np
from enclave_model import Enclave

import utils

import json
import pathlib
import os
import random

data_dir = 'data/lfw'
data_dir = pathlib.Path(data_dir)
label_file_name = os.path.join(data_dir, 'labelings.json')

with open(label_file_name, 'r') as label_file:
    labels = json.load(label_file)

all_paths = list(data_dir.glob('*/*'))
random.shuffle(all_paths)
all_images = [str(path) for path in all_paths]
all_labels = np.array([labels[path.parent.name] for path in all_paths])

# sort labels by their count
uniques, counts = np.unique(all_labels, return_counts=True)
# zip returns a generator, so it's only good for one use
zipped = zip(uniques, counts)
sorted_tuples = sorted(zipped, key=lambda x: x[1], reverse=True)

# select subset of classes, ordered by number of samples
NUM_CLASSES = 100
included_classes, _ = set(zip(*sorted_tuples[:NUM_CLASSES]))
included_images = list(
    map(lambda t: t[1],
        filter(lambda t: all_labels[t[0]] in included_classes,
               enumerate(all_images))))
included_labels = list(
    filter(lambda l: l in included_classes,
           all_labels))
# reformat labels to [0,NUM_CLASSES)
new_class_mapping = {}
for i, c in enumerate(included_classes):
    new_class_mapping[c] = i
included_labels = list(map(lambda l: new_class_mapping[l], included_labels))


data_size = len(included_images)
train_test_split = data_size - (int)(data_size*0.2)

x_train = included_images[:train_test_split]
x_test = included_images[train_test_split:]
y_train = included_labels[:train_test_split]
y_test = included_labels[train_test_split:]

IMG_SIZE = 250
BATCH_SIZE = 32

train_ds = utils.generate_dataset(
    x_train, y_train, preprocess_function=utils.preprocess_lfw)
validation_ds = utils.generate_dataset(
    x_test, y_test, preprocess_function=utils.preprocess_lfw)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model_file = 'models/vgg_lfw.h5'

VGG16_MODEL = VGG16(input_shape=IMG_SHAPE,
                    include_top=False, weights='imagenet')
VGG16_MODEL.trainable = False

hidden_neurons = 4096
steps_per_epoch = 40
num_epochs = 100

# ----------- MODEL DEFINITIONS -----------

# "baseline" variant, reaches 61%
# enclave = Enclave([
#     Dense(hidden_neurons, activation='relu'),
#     Dense(hidden_neurons, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     GlobalAveragePooling2D(),
#     enclave
# ])

# "longer" variant, with double the training epochs, reaches 62%
# num_epochs = 200
# enclave = Enclave([
#     Dense(hidden_neurons, activation='relu'),
#     Dense(hidden_neurons, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     GlobalAveragePooling2D(),
#     enclave
# ])

# "flattened" variant, with less input reduction, reaches 63%
# enclave = Enclave([
#     Dense(hidden_neurons, activation='relu'),
#     Dense(hidden_neurons, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     enclave
# ])

# "larger" variant, with more dense neurons, reaches 62%
# hidden_neurons = 8192
# enclave = Enclave([
#     Dense(hidden_neurons, activation='relu'),
#     Dense(hidden_neurons, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     GlobalAveragePooling2D(),
#     enclave
# ])

# large variant on flattened conv output
hidden_neurons = 8192
enclave = Enclave([
    Dense(hidden_neurons, activation='relu'),
    Dense(hidden_neurons, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])
dense = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    enclave
])

# ----------- COMMON AGAIN -----------
model = tf.keras.Sequential([
    VGG16_MODEL,
    dense
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs=num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=2,
                    validation_data=validation_ds)
model.save(model_file)

validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_ds, steps=validation_steps)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

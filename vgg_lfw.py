# coding: utf-8

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow as tf
from tensorflow.keras.applications import VGG16

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
all_labels = [labels[path.parent.name] for path in all_paths]

data_size = len(all_images)
train_test_split = data_size - (int)(data_size*0.2)

x_train = all_images[:train_test_split]
x_test = all_images[train_test_split]
y_train = all_labels[:train_test_split]
y_test = all_labels[train_test_split:]

IMG_SIZE = 250
BATCH_SIZE = 32

train_ds = utils.generate_dataset(
    x_train, y_train, preprocess_function=utils.preprocess_lfw)
validation_ds = utils.generate_dataset(
    x_test, y_test, preprocess_function=utils.preprocess_lfw)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model_file = 'models/vgg_lfw.h5'

VGG16_MODEL = VGG16(input_shape=(250, 250, 3),
                    include_top=False, weights='imagenet')
VGG16_MODEL.trainable = False

hidden_neurons = 4096

model = tf.keras.Sequential([
    VGG16_MODEL,
    GlobalAveragePooling2D(),
    Dense(hidden_neurons, activation='relu'),
    Dense(hidden_neurons, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(train_ds,
                    epochs=100,
                    steps_per_epoch=2,
                    validation_steps=2,
                    validation_data=validation_ds)
model.save(model_file)

validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_ds, steps=validation_steps)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
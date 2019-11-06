# coding: utf-8

import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from enclave_model import Enclave

import utils
import datasets

NUM_CLASSES = 100
x, y = datasets.build_lfw(num_classes=NUM_CLASSES, drop_max=1)

data_size = len(x)
training_split = 0.1
train_test_split = data_size - (int)(data_size*training_split)

x_train = x[:train_test_split]
x_test = x[train_test_split:]
y_train = y[:train_test_split]
y_test = y[train_test_split:]

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
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     layers.GlobalAveragePooling2D(),
#     enclave
# ])

# "shorter" variant, with half the training epochs
num_epochs = 50
hidden_neurons = 512
enclave = Enclave([
    layers.Dense(hidden_neurons, activation='relu'),
    layers.Dense(hidden_neurons, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
dense = tf.keras.Sequential([
    layers.GlobalAveragePooling2D(),
    enclave
])

# "longer" variant, with double the training epochs, reaches 62%
# num_epochs = 200
# enclave = Enclave([
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     layers.GlobalAveragePooling2D(),
#     enclave
# ])

# "flattened" variant, with less input reduction, reaches 63%
# enclave = Enclave([
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     layers.Flatten(),
#     enclave
# ])

# "larger" variant, with more dense neurons, reaches 62%
# hidden_neurons = 8192
# enclave = Enclave([
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     layers.GlobalAveragePooling2D(),
#     enclave
# ])

# large variant on flattened conv output, reaches 58%
# hidden_neurons = 8192
# enclave = Enclave([
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     layers.Flatten(),
#     enclave
# ])

# large variant on flattened conv output, longer training, reaches 65%
# hidden_neurons = 8192
# num_epochs = 200
# enclave = Enclave([
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(hidden_neurons, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     enclave
# ])

# ----------- COMMON AGAIN -----------
print('Hypeparameters:')
print('num_epochs: {}'.format(num_epochs))
print('hidden_neurons: {}'.format(hidden_neurons))
print('training set size: {}'.format(len(y_train)))
print('test set size: {}'.format(len(y_test)))
print()

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
                    validation_steps=steps_per_epoch,
                    validation_data=validation_ds)

validation_steps = steps_per_epoch
loss0, accuracy0 = model.evaluate(validation_ds, steps=validation_steps)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(model_file))
model.save(model_file)

# coding: utf-8

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from enclave_model import Enclave

import utils
import datasets

NUM_CLASSES = 100
x, y = datasets.build_lfw(num_classes=NUM_CLASSES)

data_size = len(x)
train_test_split = data_size - (int)(data_size*0.2)

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

model_file = 'models/resnet_lfw.h5'

resnet = ResNet50(input_shape=IMG_SHAPE,
                  include_top=False)
resnet.trainable = False

hidden_neurons = 4096
steps_per_epoch = 40
num_epochs = 100

# ----------- MODEL DEFINITIONS -----------

# "baseline" variant
enclave = Enclave([
    Dense(hidden_neurons, activation='relu'),
    Dense(hidden_neurons, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])
dense = tf.keras.Sequential([
    GlobalAveragePooling2D(),
    enclave
])

# "longer" variant, with double the training epochs
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

# "flattened" variant, with less input reduction
# enclave = Enclave([
#     Dense(hidden_neurons, activation='relu'),
#     Dense(hidden_neurons, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     enclave
# ])

# "larger" variant, with more dense neurons
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
# hidden_neurons = 8192
# enclave = Enclave([
#     Dense(hidden_neurons, activation='relu'),
#     Dense(hidden_neurons, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     enclave
# ])

# large variant on flattened conv output, longer training
# hidden_neurons = 8192
# num_epochs = 200
# enclave = Enclave([
#     Dense(hidden_neurons, activation='relu'),
#     Dense(hidden_neurons, activation='relu'),
#     Dense(NUM_CLASSES, activation='softmax')
# ])
# dense = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     enclave
# ])

# ----------- COMMON AGAIN -----------
model = tf.keras.Sequential([
    resnet,
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

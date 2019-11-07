import tensorflow.keras.layers as layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy

from os.path import join
import os

import utils

# global config
data_dir = 'data/mit67'
train_file = 'TrainImages.txt'
test_file = 'TestImages.txt'

# build label mapping from directories
labels = {}
i = 0
for entry in os.scandir(data_dir):
    if entry.is_dir():
        labels[entry.name] = i
        i += 1

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
MODEL_FILE = 'models/resnet_mit.h5'
HIDDEN_NEURONS = 128
NUM_EPOCHS = 100
STEPS_PER_EPOCH = len(train_labels)/32//2  # half the number of batches

resnet = ResNet50(include_top=False, weights='imagenet',
                  input_shape=((224, 224, 3)))

dense = Sequential([
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

model = Sequential([
    resnet,
    layers.Flatten(),
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
                    validation_data=test_ds)

validation_steps = STEPS_PER_EPOCH*10
loss0, accuracy0 = model.evaluate(test_ds)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(MODEL_FILE))
model.save(MODEL_FILE)

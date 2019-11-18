from tensorflow.keras.models import load_model
from enclave_model import Enclave
from enclave_layer import EnclaveLayer
import utils

import os
from os.path import join

import numpy as np

model_file = 'models/mit67_57_percent.h5'
num_images = 100

model_tf = load_model(model_file)

data_dir = 'data/mit67'
test_file = 'TestImages.txt'

# build label mapping from directories
labels = {}
i = 0
for entry in os.scandir(data_dir):
    if entry.is_dir():
        labels[entry.name] = i
        i += 1

# generate dataset
with open(join(data_dir, test_file), 'r') as f:
    test_images = [join(data_dir, l.strip()) for l in f.readlines()]
# reduce dataset size
test_images = np.random.choice(test_images, num_images)
test_labels = [labels[s.split('/')[-2]] for s in test_images]

test_ds = utils.generate_dataset(test_images, test_labels, shuffle=False, repeat=False)

# predict dataset
loss, acc = model_tf.predict(test_ds)

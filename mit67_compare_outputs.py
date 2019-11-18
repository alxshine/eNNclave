from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

from enclave_model import Enclave
from enclave_layer import EnclaveLayer
import utils

import os
from os.path import join

import json
import sys

tf.compat.v1.set_random_seed(1337)
np.random.seed(1337)

#TODO: use argparse

model_file_1 = sys.argv[1]
model_file_2 = sys.argv[2]
num_images = 100

model_1 = load_model(model_file_1, custom_objects={"EnclaveLayer": EnclaveLayer})
model_2 = load_model(model_file_2, custom_objects={"EnclaveLayer": EnclaveLayer})

data_dir = 'data/mit67'
test_file = 'TestImages.txt'

# build label mapping from directories
with open(join(data_dir, "class_labels.json"), 'r') as f:
    labels = json.load(f)

# generate dataset
with open(join(data_dir, test_file), 'r') as f:
    test_images = [join(data_dir, l.strip()) for l in f.readlines()]
# reduce dataset size
test_images = np.random.choice(test_images, num_images)
test_labels = [labels[s.split('/')[-2]] for s in test_images]

test_ds = utils.generate_dataset(test_images, test_labels, shuffle=False, repeat=False)

# predict dataset
predictions_1 = model_1.predict(test_ds)
# TODO: deal with models returning the label directly
predictions_1 = np.argmax(predictions_1, axis=1)

accuracy_1 = np.equal(predictions_1, test_labels).sum()/len(test_labels)
print("Model 1 accuracy: {}".format(accuracy_1))

predictions_2 = model_2.predict(test_ds)

accuracy_2 = np.equal(predictions_2, test_labels).sum()/len(test_labels)
print("Model 2 accuracy: {}".format(accuracy_2))

same_labels = np.equals(predictions_1, predictions_2)
print("{} of {} labels are equal".format(same_labels.sum(), len(same_labels)))

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

from enclave_model import Enclave
from enclave_layer import EnclaveLayer
import utils
import interop.pymatutil as pymatutil

import os
from os.path import join

import json
import sys
import time

tf.compat.v1.enable_eager_execution()

if len(sys.argv) < 3:
    print("Usage: {} tf_model enclave_model".format(sys.argv[0]))
    sys.exit(1)

BATCH_SIZE = 1
NUM_BATCHES = 1
SKIP_FIRST = 0

tf_model_file = sys.argv[1]
tf_model = load_model(tf_model_file)
enclave_model_file = sys.argv[2]
enclave_model = load_model(enclave_model_file, custom_objects={'EnclaveLayer': EnclaveLayer})

data_dir = 'data/mit67'
test_file = 'TestImages.txt'

print('Loading data')
# build label mapping from directories
with open(join(data_dir, "class_labels.json"), 'r') as f:
    labels = json.load(f)

# generate dataset
with open(join(data_dir, test_file), 'r') as f:
    test_images = [join(data_dir, l.strip()) for l in f.readlines()]

test_labels = [labels[s.split('/')[-2]] for s in test_images]
processed_images = np.empty((len(test_images), 224, 224, 3))
for i, image in enumerate(test_images):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (224, 224))
    processed_images[i] = image

x_test = processed_images
y_test = test_labels
all_indices = np.arange(x_test.shape[0])

total_batches = NUM_BATCHES + SKIP_FIRST
tf_times = np.empty(total_batches)
enclave_times = np.empty(total_batches)

pymatutil.initialize()

for i in range(total_batches):
    test_indices = np.random.choice(all_indices, BATCH_SIZE)
    samples = x_test[test_indices]

    print("Batch %d" % i)
    
    # predict dataset
    tf_before = time.time()
    tf_predictions = tf_model.predict(samples)
    tf_after = time.time()
    tf_times[i] = tf_after - tf_before

    enclave_before = time.time()
    enclave_predictions = enclave_model.predict(samples)
    enclave_after = time.time()
    enclave_times[i] = enclave_after - enclave_before

pymatutil.teardown()

print()
print("BATCH SIZE:\t%d" % BATCH_SIZE)
print("NUM BATCHES:\t%d" % NUM_BATCHES)
print("SKIPPING FIRST %d RESULTS" % SKIP_FIRST)
tf_times = tf_times[SKIP_FIRST:]
enclave_times = enclave_times[SKIP_FIRST:]

print()
print("Tensorflow times:")
print(tf_times)
print("Mean:\t%f" % tf_times.mean())
print("Min:\t%f" % tf_times.min())
print("Max:\t%f" % tf_times.max())

print()
print("Enclave times:")
print(enclave_times)
print("Mean:\t%f" % enclave_times.mean())
print("Min:\t%f" % enclave_times.min())
print("Max:\t%f" % enclave_times.max())

print("\nEnclave is slower than TF by a factor of %f" % (enclave_times.mean()/tf_times.mean()))

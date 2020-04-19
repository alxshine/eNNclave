from tensorflow.keras.models import load_model, Sequential
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

from rotten_tomatoes_prepare_data import load_rotten_tomatoes

BATCH_SIZE = 10000
NUM_BATCHES = 50
SKIP_FIRST = 1

if len(sys.argv) < 3:
    print("Usage: {} tf_model enclave_model".format(sys.argv[0]))
    sys.exit(1)

_, _, x_test, y_test, _ = load_rotten_tomatoes('./datasets')
all_indices = np.arange(x_test.shape[0])
print("Taking {} samples from Rotten Tomatoes sentiment test set".format(BATCH_SIZE))

tf_model = load_model(sys.argv[1])
enclave_model = load_model(sys.argv[2], custom_objects={'EnclaveLayer': EnclaveLayer})

pymatutil.initialize()

total_batches = NUM_BATCHES + SKIP_FIRST
tf_times = np.empty(total_batches)
enclave_times = np.empty(total_batches)
for i in range(total_batches):
    test_indices = np.random.choice(all_indices, BATCH_SIZE)
    samples = x_test[test_indices]
    
    print("Batch %d" % i)
    
    # predict dataset
    tf_before = time.time()
    tf_predictions = tf_model.predict(samples)
    tf_after = time.time()
    tf_labels = np.argmax(tf_predictions, axis=1)
    tf_times[i] = tf_after - tf_before

    enclave_before = time.time()
    enclave_predictions = enclave_model.predict(samples)
    enclave_after = time.time()
    enclave_labels = np.argmax(enclave_predictions, axis=1)
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

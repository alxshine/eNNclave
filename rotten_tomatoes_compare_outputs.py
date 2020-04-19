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

NUM_SAMPLES = 50

if len(sys.argv) < 3:
    print("Usage: {} tf_model enclave_model".format(sys.argv[0]))
    sys.exit(1)

_, _, x_test, y_test, _ = load_rotten_tomatoes('./datasets')
all_indices = np.arange(x_test.shape[0])
print("Taking {} samples from Rotten Tomatoes sentiment test set".format(NUM_SAMPLES))
test_indices = np.random.choice(all_indices, NUM_SAMPLES)
x_test = x_test[test_indices]
y_test = y_test[test_indices]

tf_model = load_model(sys.argv[1])
enclave_model = load_model(sys.argv[2], custom_objects={'EnclaveLayer': EnclaveLayer})

# predict dataset
print("Predicting with TF model")
tf_before = time.time()
tf_predictions = tf_model.predict(x_test)
tf_after = time.time()
tf_labels = np.argmax(tf_predictions, axis=1)
tf_time = tf_after - tf_before

tf_accuracy = np.equal(tf_labels, y_test).sum()/len(y_test)
print("Prediction took {:05f} seconds".format(tf_time))
print("TF model accuracy: {}".format(tf_accuracy))

print("Predicting with Enclave model")
enclave_before = time.time()
pymatutil.initialize()
enclave_predictions = enclave_model.predict(x_test)
pymatutil.teardown()
enclave_after = time.time()
enclave_labels = np.argmax(enclave_predictions, axis=1)
enclave_time = enclave_after - enclave_before

enclave_accuracy = np.equal(enclave_labels, y_test).sum()/len(y_test)
print("Prediction took {:05f} seconds".format(enclave_time))
print("Enclave model accuracy: {}".format(enclave_accuracy))

same_labels = np.equal(tf_labels, enclave_labels)
print("{} of {} labels are equal, slowdown factor: {:.03f}".format(same_labels.sum(), len(same_labels), enclave_time/tf_time))

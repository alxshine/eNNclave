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

NUM_SAMPLES = 1000

if len(sys.argv) < 3:
    print("Usage: {} tf_model enclave_model".format(sys.argv[0]))
    sys.exit(1)

_, _, x_test, y_test, _ = load_rotten_tomatoes('./datasets')

tf_model = load_model(sys.argv[1])
enclave_model = load_model(sys.argv[2], custom_objects={'EnclaveLayer': EnclaveLayer})

pymatutil.initialize()


num_diffs = 0
for i in range(len(y_test)):
    if i % 100 == 0:
        print("i: %d" % i)
        
    sample = x_test[i:i+1]
    
    # predict dataset
    tf_predictions = tf_model.predict(sample)
    tf_labels = np.argmax(tf_predictions, axis=1)

    enclave_predictions = enclave_model.predict(sample)
    enclave_labels = np.argmax(enclave_predictions, axis=1)

    if enclave_labels[0] != tf_labels[0]:
        print("Diff on sample %d" % i)
        num_diffs += 1

print("Number of diffs: %d, out of %d total images" % (num_diffs, len(y_test)))
pymatutil.teardown()

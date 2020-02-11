from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

import utils

import mit_prepare_data

import os
from os.path import join

import json
import sys
import time

if len(sys.argv) < 2:
    print("Usage: {} model".format(sys.argv[0]))
    sys.exit(1)

model_file = sys.argv[1]
model = load_model(model_file)

x_test, y_test = mit_prepare_data.load_test_set()

# predict dataset
print("Predicting")
before = time.time()
predictions = model.predict(x_test)
after = time.time()
predictions = np.argmax(predictions, axis=1)
total_time = after - before

accuracy = np.equal(predictions, y_test).sum()/len(y_test)
print("Prediction took {:05f} seconds".format(total_time))
print("TF model accuracy: {}".format(accuracy))

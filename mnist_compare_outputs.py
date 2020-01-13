import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf

import numpy as np
import sys
import time

import utils
import interop.pymatutil as pymatutil

from enclave_layer import EnclaveLayer

if len(sys.argv) < 3:
    print("Usage: {} regular_model enclave_model".format(sys.argv[0]))
    sys.exit(1)

NUM_IMAGES = 100

tf_model_file = sys.argv[1]
tf_model = load_model(tf_model_file)
enclave_model_file = sys.argv[2]
enclave_model = load_model(enclave_model_file, custom_objects={'EnclaveLayer': EnclaveLayer})

print("Taking {} images from MNIST test_set".format(NUM_IMAGES))
test_ds = tfds.load('mnist', split=tfds.Split.TEST, as_supervised=True)
test_ds = test_ds.map(utils.preprocess_mnist)
test_ds = test_ds.shuffle(32).take(NUM_IMAGES)
test_tuples = [(x.numpy(),y.numpy()) for x,y in test_ds]
test_images, test_labels = zip(*test_tuples)
test_images = np.array(test_images)

print("Predicting with TF model")
tf_before = time.time()
tf_predictions = tf_model.predict(test_images)
tf_after = time.time()
tf_time = tf_after - tf_before
tf_labels = np.argmax(tf_predictions, axis=1)
tf_accuracy = np.equal(tf_labels, test_labels).sum()/len(test_labels)
print("Prediction took {:05f} seconds".format(tf_time))
print("TF model accuracy: {}".format(tf_accuracy))

print("Predicting with Enclave model")
enclave_before = time.time()
pymatutil.initialize()
enclave_predictions = enclave_model.predict(test_images)
pymatutil.teardown()
enclave_after = time.time()
enclave_labels = np.argmax(enclave_predictions, axis=1) 
enclave_time = enclave_after - enclave_before
print("Prediction took {:05f} seconds".format(enclave_time))

same_labels = np.equal(tf_labels, enclave_labels)

print("{} of {} labels are equal, slowdown factor: {:.03f}".format(same_labels.sum(), len(same_labels), enclave_time/tf_time))

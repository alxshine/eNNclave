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

tf.compat.v1.enable_eager_execution()

if len(sys.argv) < 2:
    print("Usage: {} model".format(sys.argv[0]))
    sys.exit(1)

model_file = sys.argv[1]
model = load_model(model_file, custom_objects={'EnclaveLayer': EnclaveLayer})

data_dir = 'data/mit67'
test_file = 'TestImages.txt'

np.random.seed(1337)

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
y_test = np.array(test_labels)
all_indices = np.arange(x_test.shape[0])


test_indices = np.random.choice(all_indices, 1)
samples = x_test[test_indices]
correct_labels = y_test[test_indices]

# test if model has enclave part
has_enclave = any(["enclave" in l.name for l in model.layers])

if has_enclave:
    # split model into TF and enclave part
    for enclave_start,l in enumerate(model.layers):
        if "enclave" in l.name:
            break

    tf_part = Sequential(model.layers[:enclave_start])
    enclave_part = model.layers[enclave_start]
        
    before = time.time()
    pymatutil.initialize()
    after_setup = time.time()

    # predict dataset
    tf_prediction = tf_part(samples)
    after_tf = time.time()

    final_prediction = enclave_part(tf_prediction)
    after_enclave = time.time()
    
    pymatutil.teardown()
    after_teardown = time.time()

    print()
    setup_time = after_setup - before
    gpu_time = after_tf - after_setup
    enclave_time = after_enclave - after_tf
    teardown_time = after_teardown - after_enclave
    total_time = after_teardown - before
    print("Enclave setup time: {} seconds".format(setup_time))
    print("GPU time: {} seconds".format(gpu_time))
    print("Enclave time: {} seconds".format(enclave_time))
    print("Enclave teardown time: {} seconds".format(teardown_time))
    print("Total time: {} seconds".format(total_time))
    print("Difference between sum of times and total time: {}".format(total_time - (setup_time + gpu_time + enclave_time + teardown_time)))

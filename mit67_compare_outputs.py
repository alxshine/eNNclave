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

tf.compat.v1.enable_eager_execution()

if len(sys.argv) < 3:
    print("Usage: {} tf_model enclave_model".format(sys.argv[0]))

NUM_IMAGES = 20

tf_model_file = sys.argv[1]
tf_model = load_model(tf_model_file)
enclave_model_file = sys.argv[2]
enclave_model = load_model(enclave_model_file, custom_objects={'EnclaveLayer': EnclaveLayer})

data_dir = 'data/mit67'
test_file = 'TestImages.txt'

print("Taking {} images from MIT67 test_set".format(NUM_IMAGES))
# build label mapping from directories
with open(join(data_dir, "class_labels.json"), 'r') as f:
    labels = json.load(f)

# generate dataset
with open(join(data_dir, test_file), 'r') as f:
    test_images = [join(data_dir, l.strip()) for l in f.readlines()]
# reduce dataset size
test_images = np.random.choice(test_images, NUM_IMAGES)
test_labels = [labels[s.split('/')[-2]] for s in test_images]
processed_images = np.zeros((len(test_images), 224, 224, 3))
for i, image in enumerate(test_images):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (224, 224))
    processed_images[i] = image

test_images = processed_images

# predict dataset
print("Predicting with TF model")
tf_predictions = tf_model.predict(test_images)
tf_labels = np.argmax(tf_predictions, axis=1)

tf_accuracy = np.equal(tf_labels, test_labels).sum()/len(test_labels)
print("TF model accuracy: {}".format(tf_accuracy))

print("Predicting with Enclave model")
enclave_predictions = enclave_model.predict(test_images)
enclave_labels = np.argmax(enclave_predictions, axis=1)

same_labels = np.equal(tf_labels, enclave_labels)
print("{} of {} labels are equal".format(same_labels.sum(), len(same_labels)))

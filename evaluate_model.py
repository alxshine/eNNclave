from enclave_model import Enclave
from enclave_layer import EnclaveLayer
from tensorflow.keras.models import load_model
import numpy as np

import argparse
import os

parser = argparse.ArgumentParser(
    description='Evaluate TF model on test data')
parser.add_argument(
    'model_file', help='the .h5 file where the TF model is stored')
parser.add_argument(
    'data_dir',
    help='the directory where the x_test.npy and y_test.npy are stored')
parser.add_argument(
    '-n',
    dest='num_samples',
    type=int,
    default=10,
    required=False,
    help='the number of samples to evaluate')

args = parser.parse_args()
model_file = args.model_file
data_dir = args.data_dir

print('Loading model from %s' % model_file)
model = load_model(model_file, custom_objects={
                   'Enclave': Enclave, 'EnclaveLayer': EnclaveLayer})

sample_file = os.path.join(data_dir, 'x_test.npy')
print('Loading samples from %s' % sample_file)
x_test = np.load(sample_file)
label_file = os.path.join(data_dir, 'y_test.npy')
print('Loading labels from %s' % label_file)
y_test = np.load(label_file)

num_samples = args.num_samples
start = np.random.randint(x_test.shape[0]-num_samples)
test_samples = x_test[start:(start+num_samples)]
test_labels = y_test[start:(start+num_samples)]

print('Predicting on %d samples, starting from index %d' %
      (num_samples, start))

predictions = model.predict(test_samples)
if len(predictions.shape) > 0:
    predictions = predictions.argmax(axis=1)

if test_labels.shape[1] > 1:
    test_labels = test_labels.argmax(axis=1)

accuracy = np.equal(predictions, test_labels).sum()/num_samples
print('Accuracy: %f' % accuracy)

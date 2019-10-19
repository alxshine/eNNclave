from enclave_model import Enclave
from enclave_layer import EnclaveLayer
from tensorflow.keras.models import load_model
import numpy as np

import argparse
import pathlib
import random

import utils

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

print('Loading samples from %s' % data_dir)
num_samples = args.num_samples

label_names = {'daisy': 0, 'dandelion': 1,
               'roses': 2, 'sunflowers': 3, 'tulips': 4}
data_dir = pathlib.Path(data_dir)
all_images = [str(path) for path in data_dir.glob('*/*')]

print('Generating dataset of %d samples' % num_samples)
test_images = random.sample(all_images, num_samples)
test_labels = [label_names[pathlib.Path(path).parent.name]
               for path in test_images]
ds = utils.generate_dataset(
    test_images, test_labels, repeat=False, shuffle=False, batch_size=1)

print('Predicting')
predictions = model.predict_generator(ds, steps=num_samples)
if len(predictions.shape) > 0:
    predictions = predictions.argmax(axis=1)

correct_labels = [y.numpy()[0] for _,y in ds]

accuracy = np.equal(predictions, correct_labels).sum()/num_samples
print('Accuracy: %f' % accuracy)

from enclave_model import Enclave
from tensorflow.keras.models import load_model

import argparse

parser = argparse.ArgumentParser(
    description='Build C files for enclave from TF model')
parser.add_argument(
    'model_file', help='the .h5 file where the TF model is stored')
# parser.add_argument(
# 'output_dir', metavar='t', default='.', help='the output directory')

args = parser.parse_args()
model_file = args.model_file

print('Loading model from %s' % model_file)
model = load_model(model_file, custom_objects={'Enclave': Enclave})

print('Extracting enclave')
enclave = model.get_layer('Enclave')

print('Generating state files')
enclave.generate_state()
print('Generating dense()')
enclave.generate_dense()

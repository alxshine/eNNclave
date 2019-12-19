from enclave_model import Enclave
from enclave_layer import EnclaveLayer
from tensorflow.keras.models import load_model, Model, Sequential

import argparse
import pathlib

def get_all_layers(model):
    """ Get all layers of model, including ones inside a nested model """
    layers = []
    for l in model.layers:
        if hasattr(l, 'layers'):
            layers += get_all_layers(l)
        else:
            layers.append(l)
    return layers

parser = argparse.ArgumentParser(
    description='Build C files for enclave from TF model')
parser.add_argument(
    'model_file', help='the .h5 file where the TF model is stored')
parser.add_argument(
    'n', type=int, help='the number of layers to put in the enclave')
# parser.add_argument(
# 'output_dir', metavar='t', default='.', help='the output directory')

args = parser.parse_args()
model_file = args.model_file
n = args.n

print('Loading model from %s' % model_file)
model = load_model(model_file, custom_objects={'Enclave': Enclave})

# build flattened model structure
all_layers = get_all_layers(model)
num_layers = len(all_layers)

# extract the last n layers
enclave = Enclave()
for i in range(num_layers-n, num_layers):
    layer = all_layers[i]
    enclave.add(layer)

enclave_input_shape = all_layers[-n].input_shape
enclave.build(input_shape=enclave_input_shape)

# build cpp and bin files for enclave
enclave.generate_state(target_dir='lib/sgx/Enclave')
enclave.generate_forward(target_dir='lib/sgx/Enclave')

# build replacement layer for original model
enclave_model = Sequential(all_layers[:-n])
enclave_model.add(EnclaveLayer(model.layers[-1].output_shape[1]))
enclave_model.build(enclave_input_shape)
print("New model:")
enclave_model.summary()

print("Enclave:")
enclave.summary()

model_path = pathlib.Path(model_file)
target_dir = model_path.parent
target_basename = model_path.stem
target_ending = model_path.suffix

new_filename = target_basename + '_enclave' + target_ending
target_file = target_dir.joinpath(new_filename)

print('\n')
print('Saving model to {}'.format(target_file))
enclave_model.save(target_file)

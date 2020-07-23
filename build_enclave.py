from tensorflow.keras.models import load_model, Model, Sequential

from enclave_model import Enclave
from enclave_layer import EnclaveLayer
from utils import get_all_layers

import argparse
import pathlib
import subprocess
import sys

def get_new_filename(model_path):
    model_path = pathlib.Path(model_path)
    target_dir = model_path.parent
    target_basename = model_path.stem
    target_ending = model_path.suffix

    new_filename = target_basename + '_enclave' + target_ending
    target_file = target_dir.joinpath(new_filename)
    
    return target_file

def generate_enclave(enclave):
    # build cpp and bin files for enclave
    enclave.generate_state()
    enclave.generate_forward(target_dir='lib/enclave/enclave')
    enclave.generate_config(target_dir='lib/enclave/enclave/')
    # same for regular C
    enclave.generate_state()
    enclave.generate_forward(target_dir='lib/native')

def compile_enclave(verbose=False):
    if verbose:
        out=subprocess.STDOUT
        err=subprocess.STDERR
    else:
        out=subprocess.DEVNULL
        err=subprocess.DEVNULL

    make_result = subprocess.run(["make", "lib", "Build_Mode=HW_PRERELEASE"], stdout=out, stderr=err)
    if make_result.returncode != 0:
        output = ""
        if make_result.stdout is not None:
            output += make_result.stdout + "\n"
        if make_result.stderr is not None:
            output += make_result.stderr
        
        raise OSError(output)

def build_enclave(model_file, n, conn=None):
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
    generate_enclave(enclave)

    # build replacement layer for original model
    enclave_model = Sequential(all_layers[:-n])
    enclave_model.add(EnclaveLayer(model.layers[-1].output_shape[1]))
    enclave_model.build(enclave_input_shape)
    print("New model:")
    enclave_model.summary()

    print("Enclave:")
    enclave.summary()

    new_filename = get_new_filename(model_file)
    
    print('\n')
    print('Saving model to {}'.format(new_filename))
    enclave_model.save(new_filename)

    compile_enclave()
    
    print("Success!")

    return enclave_model


if __name__ == '__main__':
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

    build_enclave(model_file, n)

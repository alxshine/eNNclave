from tensorflow.keras.models import load_model

from utils import get_all_layers
from enclave_layer import EnclaveLayer
from enclave_model import Enclave

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} model_file".format(sys.argv[0]))
        sys.exit(1)
    
    model_file = sys.argv[1]
    model = load_model(model_file, custom_objects={
                       "Enclave": Enclave, "EnclaveLayer": EnclaveLayer})
    
    # go through all layers in all submodels
    all_layers = get_all_layers(model)
    lens = [len(l.name) for l in all_layers]
    max_len = max(lens)
    for i, l in enumerate(all_layers):
        try:
            shape_string = str(l.output_shape)
        except AttributeError:
            shape_string = ""

        name = l.name
        name += ":" + " " * (max_len - len(name))
        print("{}\t{:}\t{}".format(len(all_layers)-i, name, shape_string))

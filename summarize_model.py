from tensorflow.keras.models import load_model

from enclave_layer import EnclaveLayer
from enclave_model import Enclave

import sys


def get_all_layers(model):
    """ Get all layers of model, including ones inside a nested model """
    layers = []
    for l in model.layers:
        if hasattr(l, 'layers'):
            layers += get_all_layers(l)
        else:
            layers.append(l)
    return layers

def get_tikz(model):
    start_x = 0
    width = 1.8
    height = 0.4
    node_distance = 0.5

    layers = get_all_layers(model)
    ret = ''
    ret += '\\newcommand{\\netsummary}[1]{\n'
    for i,l in enumerate(layers):
        cleaned_name = l.name.replace('_','\_')
        current_x = start_x + node_distance*i
        ret += "\\node[draw=black,minimum width=%fcm,minimum height=%fcm,rotate=-90, anchor=south west] at (%f,#1) {\\tiny %s};" \
            % (width, height, current_x, cleaned_name)
        ret += "\n"
    ret += '}\n'

    ret += '\\newcommand{\\netwidth}{%f}\n' % (current_x + height)

    return ret

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} model_file".format(sys.argv[0]))
        sys.exit(1)
    
    model_file = sys.argv[1]
    model = load_model(model_file, custom_objects={
                       "Enclave": Enclave, "EnclaveLayer": EnclaveLayer})

    tikz = get_tikz(model)
    print(tikz)
    sys.exit(0)
    
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

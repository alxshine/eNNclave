from tensorflow.keras.models import load_model

from utils import get_all_layers
from enclave_layer import EnclaveLayer
from enclave_model import Enclave

import sys

def get_tikz(model):
    start_x = 0
    width = 1.8
    height = 0.4
    node_distance = 0.5
    space_between = node_distance - height

    graph_x_ticks = '\\newcommand{\\graphxticks}{'
    layers = get_all_layers(model)
    ret = ''
    ret += '\\newcommand{\\netsummary}[1]{\n'
    for i,l in enumerate(layers):
        cleaned_name = l.name.replace('_','\_')
        current_x = start_x + node_distance*i
        ret += "\\node[draw=black,minimum width=%fcm,minimum height=%fcm,rotate=-90, anchor=south west] at (%f,#1) {\\tiny %s};" \
            % (width, height, current_x, cleaned_name)
        ret += "\n"

        if i > 0:
            if i > 1:
                graph_x_ticks += ','
            graph_x_ticks += '%f' % (current_x - space_between/2)
        
    ret += '}\n'

    graph_x_ticks += '}\n'
    ret += graph_x_ticks

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

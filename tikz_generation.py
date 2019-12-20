from tensorflow.keras.models import load_model

from utils import get_all_layers
from enclave_layer import EnclaveLayer
from enclave_model import Enclave

import numpy as np

import sys
import json

def _texify_number(num):
    "uses a-j instead of 0-9 to work with tex"
    ret = ''
    num_digits = int(np.ceil(np.log10(num)))

    # TODO: ugly hacks for edge cases that I don't want to deal with right now
    if num_digits == 0:
        return 'a'
    if num == 10:
        return 'ba'
    
    for i in range(num_digits):
        int_val = ord('a') + (num % 10)
        ret += chr(int_val)
        num //= 10
    return ret[::-1]

def net_summary(model):
    start_x = 0
    width = 1.8
    height = 0.4
    node_distance = 0.5
    space_between = node_distance - height

    ret = ''
    ret += '\\newcommand{\\startx}{%f}\n' % (start_x)
    ret += '\\newcommand{\\nodedistance}{%f}\n' % (node_distance)
    ret += '\\newcommand{\\spacebetween}{%f}\n' % (space_between)
    ret += '\\newcommand{\\layerheight}{%f}\n' % (height)
    
    x_ticks = '\\newcommand{\\xticks}{'
    
    layers = get_all_layers(model)
    ret += '\n\\newcommand{\\netsummary}[1]{\n'
    for i,l in enumerate(layers):
        cleaned_name = l.name.replace('_','\_')
        current_x = start_x + node_distance*i
        ret += "\\node[draw=black,minimum width=%fcm,minimum height=%fcm,rotate=-90, anchor=south west] at (%f,#1) {\\tiny %s};" \
            % (width, height, current_x, cleaned_name)
        ret += "\n"

        if i > 0:
            if i > 1:
                x_ticks += ','
            x_ticks += '%f' % (current_x - space_between/2)

    x_ticks += '}\n'
    ret += '}\n'

    ret += x_ticks
    ret += '\\newcommand{\\netwidth}{%f}\n' % (current_x + height)

    return ret

def time_rectangles(time_dict):
    y_max = 5
    scale_bottom = -2.5 #value of -2.5 shall at 0
    scale_top = 4.7 #value of 4.7 shall be at y_max
    y_ticks = np.concatenate([np.arange(0.1, 1, 0.1), np.arange(1, 10, 1), np.arange(10, 110, 10)])

    ret = ''
    ret += '\\newcommand{\\ymax}{%f}\n' % y_max
    ret += '\\newcommand{\\yticks}{'
    for i,y in enumerate(y_ticks):
        coordinate = (np.log(y)-scale_bottom)/(scale_top-scale_bottom)
        
        if i > 0:
            ret += ','
        ret += '%f' % (coordinate*y_max)
    ret += '}\n'

    for split in time_dict:
        split = int(split)
        x_coordinate = '\\netwidth-\\layerheight-%d*\\nodedistance-\\spacebetween/2' % (split-1)
        node = '\\node[anchor=south, draw] at (%s, 0) {};' % x_coordinate

        ret += '\\newcommand{\\split%s}{%s}\n' % (_texify_number(split), node)
        
    return ret

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} model_file".format(sys.argv[0]))
        sys.exit(1)
    
    model_file = sys.argv[1]
    model = load_model(model_file, custom_objects={
                       "Enclave": Enclave, "EnclaveLayer": EnclaveLayer})

    tikz = net_summary(model)
    print(tikz)

    with open('timing_logs/mit67_times.json', 'r') as f:
        time_dict = json.load(f)
    print(time_rectangles(time_dict))

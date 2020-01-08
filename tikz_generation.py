from tensorflow.keras.models import load_model

from utils import get_all_layers
from enclave_layer import EnclaveLayer
from enclave_model import Enclave

import numpy as np
import pandas as pd

import sys
import json
import argparse
import os.path as path

# global config
Y_MAX = 7


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

def _build_log_scale(smallest_exponent, largest_exponent):
    num_scales = largest_exponent - smallest_exponent + 1
    y_ticks = np.empty(num_scales*9+1)
    y_labels = []
    for i, exp in enumerate(range(smallest_exponent, largest_exponent+1)):
        base = 10**exp
        y_labels.append((base, "10^{%d}" % exp))
        for j in range(9):
            y_ticks[i*9+j] = (j+1)*base

    y_ticks[-1] = 10*base
    y_labels.append((10*base, "10^{%d}" % (exp + 1)))

    return y_ticks, y_labels

def _calc_log_coord(lin_coord):
    scale_bottom = -3.2 #y value to be at tikz 0
    scale_top = 2.2 #y value to be at Y_MAX

    scale_width = scale_top - scale_bottom
    log_coord = np.log10(lin_coord)
    return (log_coord-scale_bottom)/scale_width

def net_summary(model, command_prefix):
    start_x = 0
    width = 1.8
    height = 0.4
    node_distance = 0.5
    space_between = node_distance - height

    ret = ''
    ret += '\\newcommand{\\%sstartx}{%f}\n' % (command_prefix, start_x)
    ret += '\\newcommand{\\%snodedistance}{%f}\n' % (command_prefix, node_distance)
    ret += '\\newcommand{\\%sspacebetween}{%f}\n' % (command_prefix, space_between)
    ret += '\\newcommand{\\%slayerheight}{%f}\n' % (command_prefix, height)
    
    x_ticks = '\\newcommand{\\%sxticks}{' % (command_prefix)
    
    layers = get_all_layers(model)
    ret += '\n\\newcommand{\\%snetsummary}[1]{\n' % (command_prefix)  
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

def generate_y_axis():
    y_ticks, y_labels = _build_log_scale(-3, 1)
    ret = ''
    ret += '\\newcommand{\\ymax}{%f}\n' % Y_MAX
    ret += '\\newcommand{\\yticks}{'
    with np.errstate(all='raise'):
        try:
            for i,y in enumerate(y_ticks):
                coordinate = _calc_log_coord(y)
            
                if i > 0:
                    ret += ','
                ret += '%f' % (coordinate*Y_MAX)
        except FloatingPointError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            print("y: %f" % y, file=sys.stderr)

    ret += '}\n'

    # generate command for y labels
    label_command = "\\newcommand{\\ylabels}[1]{\n"
    for lin_y, label_text in y_labels:
        log_y = _calc_log_coord(lin_y)

        current_label = "\\draw (#1, %f) node {\\tiny $%s$};\n" % (log_y * Y_MAX, label_text)
        label_command += current_label
    label_command += "}\n"

    ret += label_command

    return ret


def time_rectangles(times, command_prefix):
    rectangle_width = 0.15

    ret = ''

    # generate time rectangles
    for i, row in times.iterrows():
        gpu_time = row['gpu_time']
        enclave_time = row['combined_enclave_time']
        native_time = row['native_time']
        split = int(row['layers_in_enclave'])
        
        left_0 = '\\netwidth-\\layerheight-%d*\\nodedistance-\\spacebetween/2-%f' % (split-1, rectangle_width*3/2)
        right_0 = left_0 + ("+%f" % rectangle_width)
        right_1 = left_0 + ("+%f" % (2*rectangle_width))
        right_2 = left_0 + ("+%f" % (3*rectangle_width))
        
        with np.errstate(all='raise'):
            try:
                gpu_north = _calc_log_coord(gpu_time)*Y_MAX
                native_north = _calc_log_coord(native_time)*Y_MAX
                enclave_north = _calc_log_coord(enclave_time)*Y_MAX
            except FloatingPointError as e:
                print("ERROR: %s" % e, file=sys.stderr)
                print("GPU time: %f, native time: %f, enclave time: %f" % (gpu_time, native_time, enclave_time), file=sys.stderr)
        
        node = '\\draw[fill=color1] (%s, 0) rectangle (%s, %f);\n' % (left_0, right_0, gpu_north)
        node += '\\draw[fill=color4] (%s, 0) rectangle (%s, %f);\n' % (right_0, right_1, native_north)
        node += '\\draw[fill=color7] (%s, 0) rectangle (%s, %f);\n' % (right_1, right_2, enclave_north)

        ret += '\\newcommand{\\%ssplit%s}{%s}\n' % (command_prefix, _texify_number(split), node)

        
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate tikz graphics')
    parser.add_argument('time_files', metavar='time_file', type=str, nargs='+',
            help='a time file to load generate a graphic for')
    parser.add_argument('--model', dest='model_files', metavar='model_file', type=str, nargs='+',
            help='a model file to generate a summary for')
    parser.add_argument('--y_axis', action='store_true',
            help='generate macro for y axis ticks and labels')

    args = parser.parse_args()

    for f in args.time_files:
        times = pd.read_csv(f)
        basename = path.basename(f)
        without_extension,_ = path.splitext(basename)
        parts = without_extension.split('_')
        device = parts[-1]
        model_name = parts[0]
        command_prefix = model_name + device
        print(time_rectangles(times, command_prefix))

    if args.model_files:
        for f in args.model_files:
            basename = path.basename(f)
            without_extension,_ = path.splitext(basename)
            model_name = without_extension.split('_')[0]
            model = load_model(f, custom_objects={'EnclaveLayer': EnclaveLayer, 'Enclave': Enclave})
            print(net_summary(model, model_name))

    if args.y_axis:
        print(generate_y_axis())

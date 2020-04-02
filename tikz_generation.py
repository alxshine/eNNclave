from tensorflow.keras.models import load_model

from utils import get_all_layers, get_dataset_from_model_path
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

    if num == 0:
        return 'a'
    
    #  for i in range(num_digits):
    while num > 0:
        int_val = ord('a') + (num % 10)
        ret += chr(int_val)
        num //= 10
        
        #  if num == 0:
            #  ret += 'a'

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

def _calc_log_coord(lin_coord, y_min, y_max):
    scale_width = (y_max+0.05) - (y_min-0.05)
    log_coord = np.log10(lin_coord)
    return (log_coord-y_min)/scale_width

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
    i = 1
    for l in reversed(layers):
        if 'input' in l.name or 'embedding' in l.name:
            continue
        
        cleaned_name = l.name.replace('_','\_')
        current_x = start_x + node_distance*i
        ret += "\\node[draw=black,minimum width=%fcm,minimum height=%fcm,rotate=-90, anchor=south west] at (%f,#1) {\\tiny %s};" \
            % (width, height, current_x, cleaned_name)
        ret += "\n"

        if i > 1:
            x_ticks += ','
        x_ticks += '%f' % (current_x - space_between/2)

        i += 1

    x_ticks += ',' + '%f' % (start_x + node_distance*i - space_between/2)
    x_ticks += '}\n'
    ret += '}\n'

    ret += x_ticks
    ret += '\\newcommand{\\netwidth}{%f}\n' % (current_x + height)

    return ret

def generate_y_axis(y_min, y_max):
    y_ticks, y_labels = _build_log_scale(y_min, y_max-1) # TODO: ugly workaround for scales
    ret = ''
    ret += '\\newcommand{\\ymax}{%f}\n' % Y_MAX
    ret += '\\newcommand{\\yticks}{'
    with np.errstate(all='raise'):
        try:
            for i,y in enumerate(y_ticks):
                coordinate = _calc_log_coord(y, y_min, y_max)
            
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
        log_y = _calc_log_coord(lin_y, y_min, y_max)

        current_label = "\\draw (#1, %f) node {\\tiny $%s$};\n" % (log_y * Y_MAX, label_text)
        label_command += current_label
    label_command += "}\n"

    ret += label_command

    return ret


def time_rectangles(times, platform, y_min, y_max):
    rectangle_width = 0.3

    ret = ''

    # generate time rectangles
    for i, row in times.iterrows():
        tf_time = row['tf_time']
        enclave_time = row['combined_enclave_time']
        native_time = row['native_time']
        split = int(row.name)
        
        left_0 = f"{split+1}*\\nodedistance - \\spacebetween/2 - {rectangle_width/2}"
        right_0 = left_0 + ("+%f" % rectangle_width)
        right_1 = left_0 + ("+%f" % (2*rectangle_width))
        
        with np.errstate(all='raise'):
            try:
                tf_north = _calc_log_coord(tf_time, y_min, y_max)*Y_MAX
                native_north = _calc_log_coord(tf_time + native_time, y_min, y_max)*Y_MAX
                enclave_north = _calc_log_coord(tf_time + native_time + enclave_time, y_min, y_max)*Y_MAX
            except FloatingPointError as e:
                print("ERROR: %s" % e, file=sys.stderr)
                print("GPU time: %f, native time: %f, enclave time: %f" % (tf_time, native_time, enclave_time), file=sys.stderr)
        
        node = '\\draw[fill=color1] (%s, 0) rectangle (%s, %f);\n' % (left_0, right_0, tf_north)
        node += '\\draw[fill=color4] (%s, %s) rectangle (%s, %f);\n' % (left_0, tf_north, right_0, native_north)
        node += '\\draw[fill=color7] (%s, %s) rectangle (%s, %f);\n' % (left_0, native_north, right_0, enclave_north)

        ret += '\\newcommand{\\%ssplit%s}{%s}\n' % (platform, _texify_number(split), node)

        
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate tikz graphics')
    parser.add_argument('--time', dest='time_files', metavar='time_file', type=str, nargs='+',
            help='a time file to load generate a graphic for')
    parser.add_argument('--model', dest='model_files', metavar='model_file', type=str, nargs='+',
            help='a model file to generate a summary for')
    parser.add_argument('--ymin', default=-1, type=int, help='Set Y minimum')
    parser.add_argument('--ymax', default=3, type=int, help='Set Y maximum')

    args = parser.parse_args()

    if args.time_files:
        for f in args.time_files:
            times = pd.read_csv(f)
            times = times.groupby(['layers_in_enclave']).mean()
            basename = path.basename(f)
            without_extension,_ = path.splitext(basename)
            parts = without_extension.split('_')
            device = parts[-1]
            print(time_rectangles(times, device, args.ymin, args.ymax))

    if args.model_files:
        for f in args.model_files:
            dataset = get_dataset_from_model_path(f)
            model = load_model(f, custom_objects={'EnclaveLayer': EnclaveLayer, 'Enclave': Enclave})
            print(net_summary(model))

        print(generate_y_axis(args.ymin, args.ymax))

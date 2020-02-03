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

def _get_constant_dict(model_name):
    template_start_x = '\\%sstartx'
    template_node_distance = '\\%snodedistance'
    template_space_between = '\\%sspacebetween'
    template_layer_height = '\\%slayerheight'
    template_net_width = '\\%snetwidth'

    return {
            'start_x': template_start_x % model_name,
            'node_distance': template_node_distance % model_name,
            'space_between': template_space_between % model_name,
            'layer_height': template_layer_height % model_name,
            'net_width': template_net_width % model_name
            }


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

def _calc_log_coord(lin_coord):
    scale_bottom = -2.1 #y value to be at tikz 0
    scale_top = 3.2 #y value to be at Y_MAX

    scale_width = scale_top - scale_bottom
    log_coord = np.log10(lin_coord)
    return (log_coord-scale_bottom)/scale_width

def net_summary(model, model_name):
    start_x = 0
    width = 1.8
    height = 0.4
    node_distance = 0.5
    space_between = node_distance - height
    constant_dict = _get_constant_dict(model_name)

    ret = ''
    ret += '\\newcommand{%s}{%f}\n' % (constant_dict['start_x'], start_x)
    ret += '\\newcommand{%s}{%f}\n' % (constant_dict['node_distance'], node_distance)
    ret += '\\newcommand{%s}{%f}\n' % (constant_dict['space_between'], space_between)
    ret += '\\newcommand{%s}{%f}\n' % (constant_dict['layer_height'], height)
    
    x_ticks = '\\newcommand{\\%sxticks}{' % (model_name)
    
    layers = get_all_layers(model)
    ret += '\n\\newcommand{\\%snetsummary}[1]{\n' % (model_name)
    i = 1
    for l in layers:
        if 'input' in l.name:
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

    x_ticks += '}\n'
    ret += '}\n'

    ret += x_ticks
    ret += '\\newcommand{%s}{%f}\n' % (constant_dict['net_width'], current_x + height)

    return ret

def generate_y_axis(model_name):
    y_ticks, y_labels = _build_log_scale(-1, 2)
    ret = ''
    ret += '\\newcommand{\\%symax}{%f}\n' % (model_name, Y_MAX)
    ret += '\\newcommand{\\%syticks}{' % (model_name)
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


def time_rectangles(times, model_name, platform):
    rectangle_width = 0.3
    constant_dict = _get_constant_dict(model_name)

    ret = ''

    # generate time rectangles
    for i, row in times.iterrows():
        tf_time = row['tf_time']
        enclave_time = row['combined_enclave_time']
        native_time = row['native_time']
        split = int(row.name)
        
        left_0 = '%s - %s - %d*%s - %s/2 - %f' % (constant_dict['net_width'], constant_dict['layer_height'],
                split-1, constant_dict['node_distance'], constant_dict['space_between'], rectangle_width/2)
        right_0 = left_0 + ("+%f" % rectangle_width)
        right_1 = left_0 + ("+%f" % (2*rectangle_width))
        
        with np.errstate(all='raise'):
            try:
                tf_north = _calc_log_coord(tf_time)*Y_MAX
                native_north = _calc_log_coord(tf_time + native_time)*Y_MAX
                enclave_north = _calc_log_coord(tf_time + native_time + enclave_time)*Y_MAX
            except FloatingPointError as e:
                print("ERROR: %s" % e, file=sys.stderr)
                print("GPU time: %f, native time: %f, enclave time: %f" % (tf_time, native_time, enclave_time), file=sys.stderr)
        
        node = '\\draw[fill=color1] (%s, 0) rectangle (%s, %f);\n' % (left_0, right_0, tf_north)
        node += '\\draw[fill=color4] (%s, %s) rectangle (%s, %f);\n' % (left_0, tf_north, right_0, native_north)
        node += '\\draw[preaction={fill,color7}, pattern=north east lines] (%s, %s) rectangle (%s, %f);\n' % (left_0, native_north, right_0, enclave_north)

        ret += '\\newcommand{\\%ssplit%s}{%s}\n' % (model_name+platform, _texify_number(split), node)

        
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate tikz graphics')
    parser.add_argument('--time', dest='time_files', metavar='time_file', type=str, nargs='+',
            help='a time file to load generate a graphic for')
    parser.add_argument('--model', dest='model_files', metavar='model_file', type=str, nargs='+',
            help='a model file to generate a summary for')
    parser.add_argument('--y_axis', action='store_true',
            help='generate macro for y axis ticks and labels')

    args = parser.parse_args()

    if args.time_files:
        for f in args.time_files:
            times = pd.read_csv(f)
            times = times.groupby(['layers_in_enclave']).mean()
            basename = path.basename(f)
            without_extension,_ = path.splitext(basename)
            parts = without_extension.split('_')
            device = parts[-1]
            model_name = parts[0]
            print(time_rectangles(times, model_name, device))

    if args.model_files:
        for f in args.model_files:
            dataset = get_dataset_from_model_path(f)
            model = load_model(f, custom_objects={'EnclaveLayer': EnclaveLayer, 'Enclave': Enclave})
            print(net_summary(model, model_name))

        if args.y_axis:
            print(generate_y_axis(model_name))

import numpy as np
import tensorflow as tf

import sys


def dump_array(name, a):
    print("float %s[] = {" % name, end='')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            print(f'{a[i,j]:.03}', end='')
            if i < a.shape[0]-1 or j < a.shape[1]-1:
                print(',', end='')
        print()
    print('};')


def dump_array_flatten(name, a):
    print("float %s[] = {" % name, end='')
    array = a.flatten()
    for i in range(array.shape[0]):
        print(f'{array[i]:.03}', end='')
        if i < array.shape[0]-1:
            print(',', end='')
        # print()
    print('};')


def generate_mul():
    rand_a = np.random.rand(3, 3) - .5
    dump_array('rand_a', rand_a)
    rand_b = np.random.rand(3, 3) - .5
    dump_array('rand_b', rand_b)
    rand_res = rand_a*rand_b
    dump_array('rand_exp', rand_res)


def generate_add():
    rand_a = np.random.rand(3, 3) - .5
    dump_array('rand_a', rand_a)
    rand_b = np.random.rand(3, 3) - .5
    dump_array('rand_b', rand_b)
    rand_res = rand_a+rand_b
    dump_array('rand_exp', rand_res)


def generate_sep_conv1(steps=3, channels=3, filters=3, kernel_size=2, mode='full'):
    print(f"int steps = {steps};")
    print(f"int channels = {channels};")
    print(f"int filters = {filters};")
    print(f"int kernel_size = {kernel_size};")
    print()

    inputs = np.random.rand(1, steps, channels)
    dump_array_flatten('inputs', inputs)

    if mode == 'zeros':
        layer = tf.keras.layers.SeparableConv1D(
            filters, kernel_size, strides=1, input_shape=inputs.shape, padding='same', use_bias=True, bias_initializer='zeros', depthwise_initializer='zeros', pointwise_initializer='zeros')
    elif mode == 'full':
        layer = tf.keras.layers.SeparableConv1D(
            filters, kernel_size, strides=1, input_shape=inputs.shape, padding='same', use_bias=True, bias_initializer='glorot_uniform', depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform')
    else:
        print("Unknown test mode")
        sys.exit(1)
    results = layer(inputs).numpy()
    params = layer.get_weights()
    depth_kernels = params[0]
    dump_array_flatten('depth_kernels', depth_kernels)
    point_kernels = params[1]
    dump_array_flatten('point_kernels', point_kernels)
    biases = params[2]
    dump_array_flatten('biases', biases)
    dump_array_flatten('expected', results)


if __name__ == "__main__":
    generate_sep_conv1(steps=10, channels=4, filters=5, kernel_size=3,mode='full')

import numpy as np
import tensorflow as tf

import sys

rng = np.random.default_rng()


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


def generate_mul(h=3, w=3, mode='full'):
    print(f"int h = {h};")
    print(f"int w = {w};")

    if mode == 'full':
        a = rng.uniform(-1, 1, (h, w))
        b = rng.uniform(-1, 1, (h, w))
    elif mode == 'zeros':
        a = rng.uniform(-1, 1, (h, w))
        b = np.zeros((h, w))
    elif mode == 'sequential':
        a = np.arange(h*w, dtype=np.float).reshape((h, w))
        b = np.ones((h, w))
    else:
        print("Unknown tests mode")
        sys.exit(1)

    dump_array_flatten('a', a)
    dump_array_flatten('b', b)
    expected = np.matmul(a,b)
    dump_array_flatten('expected', expected)


def generate_add():  # TODO: rebuild to use tensorflow
    rand_a = np.random.rand(3, 3) - .5
    dump_array_flatten('rand_a', rand_a)
    rand_b = np.random.rand(3, 3) - .5
    dump_array_flatten('rand_b', rand_b)
    rand_res = rand_a+rand_b
    dump_array_flatten('rand_exp', rand_res)


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
        print("Unknown tests mode")
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


def generate_conv2(h=3, w=3, channels=3, filters=3, kernel_size=3, mode='full'):
    print(f"int h = {h};")
    print(f"int w = {w};")
    print(f"int channels = {channels};")
    print(f"int filters = {filters};")
    print(f"int kernel_size = {kernel_size};")
    print()

    inputs = np.random.rand(1, h, w, channels)

    if mode == 'zeros':
        layer = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=1, input_shape=inputs.shape, padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer='zeros')
    elif mode == 'sequential':
        inputs = np.arange(
            h*w*channels, dtype=np.float).reshape((1, h, w, channels))
        layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, input_shape=inputs.shape,
                                       padding='same', use_bias=True, bias_initializer='zeros', kernel_initializer='ones')
    elif mode == 'full':
        layer = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=1, input_shape=inputs.shape, padding='same', use_bias=True, bias_initializer='glorot_uniform', kernel_initializer='glorot_uniform')
    else:
        print("Unknown tests mode")
        sys.exit(1)

    dump_array_flatten('inputs', inputs)

    results = layer(inputs).numpy()
    params = layer.get_weights()
    kernels = params[0]
    dump_array_flatten('kernels', kernels)
    biases = params[1]
    dump_array_flatten('biases', biases)
    dump_array_flatten('expected', results)

def generate_depthwise_conv2(h=3, w=3, channels=3, kernel_size=3, mode='full'):
    print(f"int h = {h};")
    print(f"int w = {w};")
    print(f"int channels = {channels};")
    print(f"int kernel_size = {kernel_size};")
    print()

    inputs = np.random.rand(1, h, w, channels)

    if mode == 'zeros':
        layer = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same', use_bias=False, bias_initializer='zeros', kernel_initializer='zeros')
    elif mode == 'sequential':
        inputs = np.arange(
            h*w*channels, dtype=np.float).reshape((1, h, w, channels))
        layer = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same', use_bias=False, bias_initializer='zeros', kernel_initializer='ones')
    elif mode == 'full':
        layer = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same', use_bias=False, bias_initializer='glorot_uniform', kernel_initializer='glorot_uniform')
    else:
        print("Unknown tests mode")
        sys.exit(1)

    dump_array_flatten('inputs', inputs)

    results = layer(inputs).numpy()
    params = layer.get_weights()
    kernels = params[0]
    dump_array_flatten('kernels', kernels)
    dump_array_flatten('expected', results)


def generate_relu(size=10):
    print(f"int size = {size};")
    print()

    inputs = np.random.rand(1, size)
    dump_array_flatten('inputs', inputs)

    layer = tf.keras.layers.ReLU(input_shape=inputs)
    results = layer(inputs).numpy()
    dump_array_flatten('expected', results)
    print()
    dump_array_flatten('ret', inputs)


def generate_global_average_pooling_1d(steps=10, channels=3):
    print(f"int steps = {steps};")
    print(f"int channels = {channels};")
    print()

    inputs = rng.uniform(-1, 1, (1, steps, channels))
    dump_array_flatten('inputs', inputs)

    layer = tf.keras.layers.GlobalAveragePooling1D(
        input_shape=(steps, channels))

    results = layer(inputs).numpy()
    dump_array_flatten('expected', results)


def generate_global_average_pooling_2d(h=5, w=5, channels=3):
    print(f"int h = {h};")
    print(f"int w = {w};")
    print(f"int channels = {channels};")
    print()

    inputs = rng.uniform(-1, 1, (1, h, w, channels))
    dump_array_flatten('inputs', inputs)

    layer = tf.keras.layers.GlobalAveragePooling2D(
        input_shape=(h, w, channels))

    results = layer(inputs).numpy()
    dump_array_flatten('expected', results)


def generate_max_pooling1d(steps=10, channels=3, pool_size=3, mode='random'):
    print(f"int steps = {steps};")
    print(f"int channels = {channels};")
    print(f"int pool_size = {pool_size};")
    print()

    if mode == 'random':
        inputs = rng.uniform(-1, 1, (1, steps, channels))
    elif mode == 'sequential':
        inputs = np.arange(
            steps*channels).reshape((1, steps, channels)).astype(np.float)
    else:
        print(f"Unknown tests mode {mode}")
        sys.exit(1)

    dump_array_flatten('inputs', inputs)

    layer = tf.keras.layers.MaxPooling1D(
        input_shape=inputs.shape, pool_size=pool_size, padding='valid')

    results = layer(inputs).numpy()
    dump_array_flatten('expected', results)


def generate_max_pooling2d(h=5, w=5, channels=3, pool_size=3, mode='random'):
    print(f"int h = {h};")
    print(f"int w = {w};")
    print(f"int channels = {channels};")
    print(f"int pool_size = {pool_size};")
    print()

    if mode == 'random':
        inputs = rng.uniform(-1, 1, (1, h, w, channels))
    elif mode == 'sequential':
        inputs = np.arange(
            h*w*channels, dtype=np.float).reshape(1, h, w, channels)
    else:
        print(f"Unknown tests mode {mode}")
        sys.exit(1)

    dump_array_flatten('inputs', inputs)

    layer = tf.keras.layers.MaxPooling2D(
        input_shape=inputs.shape, pool_size=pool_size, padding='valid')

    results = layer(inputs).numpy()
    dump_array_flatten('expected', results)


def generate_zero_pad2(h=3, w=3, channels=3, top_pad=1, bottom_pad=1, left_pad=1, right_pad=1):
    print(f"int h = {h};")
    print(f"int w = {w};")
    print(f"int channels = {channels};")
    print(f"int top_pad = {top_pad};")
    print(f"int bottom_pad = {bottom_pad};")
    print(f"int left_pad = {left_pad};")
    print(f"int right_pad = {right_pad};")
    print()

    inputs = rng.uniform(-1, 1, (1, h, w, channels))
    dump_array_flatten('inputs', inputs)

    layer = tf.keras.layers.ZeroPadding2D(input_shape=inputs.shape, padding=(
        (top_pad, bottom_pad), (left_pad, right_pad)))

    results = layer(inputs).numpy()
    dump_array_flatten('expected', results)


if __name__ == "__main__":
    generate_depthwise_conv2(h=3, w=3, channels=3)
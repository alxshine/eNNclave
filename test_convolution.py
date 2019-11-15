import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np

tf.compat.v1.enable_eager_execution()

input_shape = (1, 3, 3, 1)
inputs = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
conv_layer = layers.Conv2D(
    1, 3, strides=1, input_shape=inputs.shape[1:], kernel_initializer='ones', padding='same')

results = conv_layer(inputs)
weights = conv_layer.get_weights()[0]
bias = conv_layer.get_weights()[1]

print("Input shape: {}".format(inputs.shape))
print("Output shape: {}".format(results.shape))

print("Inputs:\n{}".format(inputs[0, :, :, 0]))
print("Output:\n{}".format(results[0, :, :, 0]))

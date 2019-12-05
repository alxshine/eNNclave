import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np

tf.compat.v1.enable_eager_execution()

input_shape = (1, 3, 2)
inputs = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
conv_layer = layers.SeparableConv1D(
    1,2, strides=1, input_shape=inputs.shape[1:], depthwise_initializer='ones', pointwise_initializer='ones', padding='same')

results = conv_layer(inputs)
depthwise = conv_layer.get_weights()[0]
pointwise = conv_layer.get_weights()[1]
bias = conv_layer.get_weights()[2]

conv_layer.set_weights([np.array([2,2,2,2]).reshape(depthwise.shape), np.array([1,0]).reshape(pointwise.shape), bias])
results = conv_layer(inputs)
depthwise = conv_layer.get_weights()[0]
pointwise = conv_layer.get_weights()[1]
bias = conv_layer.get_weights()[2]

print("Inputs:")
inputs = inputs[0]
print(inputs.shape)
print(inputs)

print()
    
print("Depthwise:")
print(depthwise.shape)
for fi in range(depthwise.shape[2]):
    print("fi={}".format(fi))
    print(depthwise[:,:,fi])

print()

print("Pointwise:")
print(pointwise.shape)
for fi in range(pointwise.shape[2]):
    print("fi={}".format(fi))
    print(pointwise[:,:,fi])

print()

print("Bias:")
print(bias.shape)
print(bias)

print()

print("Output:")
print(results.shape)
print(results)

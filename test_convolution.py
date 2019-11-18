import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np

tf.compat.v1.enable_eager_execution()

input_shape = (1, 3, 3, 2)
inputs = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
conv_layer = layers.Conv2D(
    2, 3, strides=1, input_shape=inputs.shape[1:], kernel_initializer='ones', padding='same')

results = conv_layer(inputs)
weights = conv_layer.get_weights()[0]
bias = conv_layer.get_weights()[1]

print("Input shape: {}".format(inputs.shape))
print("Output shape: {}".format(results.shape))

print("Inputs:")
for i in range(inputs.shape[-1]):
    print("ci={}".format(i))
    print(inputs[0,:,:,i])

print("Weights:")
for i in range(weights.shape[-2]):
    print("c={}".format(i))
    for j in range(weights.shape[-1]):
        print("f={}".format(j))
        print(weights[:,:,i,i])

print("Output:")
for i in range(results.shape[-1]):
    print("ci={}".format(i))
    print(results[0,:,:,i].numpy())

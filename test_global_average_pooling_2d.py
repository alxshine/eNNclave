import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np

tf.compat.v1.enable_eager_execution()

input_shape = (1, 3, 3, 2)
inputs = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
pool_layer = layers.GlobalAveragePooling2D()

results = pool_layer(inputs)

print("Input shape: {}".format(inputs.shape))
print("Output shape: {}".format(results.shape))

print("Inputs:")
for i in range(inputs.shape[-1]):
    print("ci={}".format(i))
    print(inputs[0,:,:,i])

print("Output:")
print(results.numpy())

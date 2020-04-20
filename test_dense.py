import tensorflow.keras.layers as layers
import tensorflow as tf

import numpy as np

tf.compat.v1.enable_eager_execution()

inputs=np.arange(3).reshape((1,3))
dense_layer = layers.Dense(3,input_shape=(len(inputs),),kernel_initializer='ones',bias_initializer='ones')
results = dense_layer(inputs).numpy()

dense_layer.set_weights([np.array([[1,1,.5],[.25,1,1],[1,1,.5]]),np.array([1,1,1])])

results = dense_layer(inputs).numpy()
weights = dense_layer.get_weights()[0]
bias = dense_layer.get_weights()[1]

print("Input shape: {}".format(inputs.shape))
print("Output shape: {}".format(results.shape))

print("Inputs:")
print(inputs)

print("Weights:")
print(weights)

print("Biases:")
print(bias)

print("Output:")
print(results)

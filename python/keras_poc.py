import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer
from tensorflow.keras import Model

import interop.pymatutil as pymatutil

tf.enable_eager_execution()

MODEL_FILE = 'models/mnist_tf_cnn.pt'

MNIST = tf.keras.datasets.mnist


def external_func(x):  # x should be numpy array
    x = x.numpy()
    b = np.ones_like(x)*42
    xb = x.astype(np.float32).tobytes()
    bb = b.astype(np.float32).tobytes()
    retb = pymatutil.add(
        xb, x.shape[0], x.shape[1], bb, b.shape[0], b.shape[1])
    return np.frombuffer(retb, dtype=np.float32).reshape(x.shape)


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.py_function(func=external_func, inp=[x], Tout=tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        self.external = MyLayer(10)
        
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.external(x)


# Create an instance of the model
model = MyModel()

pymatutil.initialize()

(x_train, y_train), (x_test, y_test) = MNIST.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# pymatutil.teardown()

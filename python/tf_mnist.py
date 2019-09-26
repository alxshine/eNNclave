import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

MODEL_FILE = 'models/mnist_tf_cnn.pt'

MNIST = tf.keras.datasets.mnist


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(np.zeros_like(x))
        x = self.d2(x)
        return x

# Create an instance of the model
model = MyModel()

(x_train, y_train), (x_test, y_test) = MNIST.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)

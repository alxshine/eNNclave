from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import numpy as np

import unittest
from .common import common_test_basis


class MaxPool1D(TensorFlowTestCase):
    def testSmall(self):
        model = Sequential([
            layers.MaxPool1D(pool_size=3, input_shape=(5, 3))
        ])
        common_test_basis(model, False)

    def testMedium(self):
        model = Sequential([
            layers.MaxPool1D(pool_size=3, input_shape=(50, 3))
        ])
        common_test_basis(model, False)

    def testLarge(self):
        model = Sequential([
            layers.MaxPool1D(pool_size=5, input_shape=(500, 10))
        ])
        common_test_basis(model, False)

    def testHuge(self):
        model = Sequential([
            layers.MaxPool1D(pool_size=64, input_shape=(1000, 1000))
        ])
        common_test_basis(model, False)

if __name__ == "__main__":
    unittest.main()
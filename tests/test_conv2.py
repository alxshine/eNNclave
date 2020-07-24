from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import numpy as np

import unittest
from .common import common_test_basis


class Conv2dTests(TensorFlowTestCase):
    def testSmall(self):
        model = Sequential([
            layers.Conv2D(3, 3, input_shape=(5, 5, 3), padding='same')
        ])
        common_test_basis(model, False)

    def testMedium(self):
        model = Sequential([
            layers.Conv2D(5, 3, input_shape=(10, 10, 5), padding='same')
        ])
        common_test_basis(model, False)

    def testLarge(self):
        model = Sequential([
            layers.Conv2D(10, 4, input_shape=(100, 100, 5), padding='same')
        ])
        common_test_basis(model, False)

    def testHuge(self):
        raise AssertionError("Causes a segfault in C")

        model = Sequential([
            layers.Conv2D(64, 5, input_shape=(500, 500, 64), padding='same')
        ])
        common_test_basis(model, False)


if __name__ == "__main__":
    unittest.main()
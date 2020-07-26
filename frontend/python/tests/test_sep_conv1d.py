from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers

import numpy as np
import os
import unittest

from .common import common_test_basis


class SepConv1dTest(TensorFlowTestCase):
    def testSmallNative(self):
        model = Sequential([
            layers.SeparableConv1D(3, 3, input_shape=(5, 3), padding='same')
        ])
        common_test_basis(model, False)

    def testMediumNative(self):
        model = Sequential([
            layers.SeparableConv1D(3, 3, input_shape=(10, 5), padding='same')
        ])
        common_test_basis(model, False)

    def testLargeNative(self):
        model = Sequential([
            layers.SeparableConv1D(10, 5, input_shape=(100, 5), padding='same')
        ])
        common_test_basis(model, False)

    def testHugeNative(self):
        model = Sequential([
            layers.SeparableConv1D(
                64, 10, input_shape=(500, 64), padding='same')
        ])
        common_test_basis(model, False)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testSmallEnclave(self):
        model = Sequential([
            layers.SeparableConv1D(3, 3, input_shape=(5, 3), padding='same')
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testMediumEnclave(self):
        model = Sequential([
            layers.SeparableConv1D(3, 3, input_shape=(10, 5), padding='same')
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testLargeEnclave(self):
        model = Sequential([
            layers.SeparableConv1D(10, 5, input_shape=(100, 5), padding='same')
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testHugeEnclave(self):
        model = Sequential([
            layers.SeparableConv1D(
                64, 10, input_shape=(500, 64), padding='same')
        ])
        common_test_basis(model, True)


if __name__ == "__main__":
    unittest.main()

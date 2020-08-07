from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers

import os
import unittest

from frontend.python.common import common_test_basis


class MaxPooling2D(TensorFlowTestCase):
    def testSmallNative(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=3, input_shape=(5, 5, 3))
        ])
        common_test_basis(model, False)

    def testMediumNative(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=3, input_shape=(10, 10, 5))
        ])
        common_test_basis(model, False)

    def testLargeNative(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=5, input_shape=(100, 100, 10))
        ])
        common_test_basis(model, False)

    @unittest.skip
    def testHugeNative(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=10, input_shape=(1000, 1000, 64))
        ])
        common_test_basis(model, False)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testSmallEnclave(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=3, input_shape=(5, 5, 3))
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testMediumEnclave(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=3, input_shape=(10, 10, 5))
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testLargeEnclave(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=5, input_shape=(100, 100, 10))
        ])
        common_test_basis(model, True)

    @unittest.skip
    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testHugeEnclave(self):
        model = Sequential([
            layers.MaxPooling2D(pool_size=10, input_shape=(1000, 1000, 64))
        ])
        common_test_basis(model, True)

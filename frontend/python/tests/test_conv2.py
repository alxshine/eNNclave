from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers

import os

import unittest
from frontend.python.common import common_test_basis


class Conv2dTests(TensorFlowTestCase):
    def testSmallNative(self):
        model = Sequential([
            layers.Conv2D(3, 3, input_shape=(5, 5, 3), padding='same')
        ])
        common_test_basis(model, False)

    def testMediumNative(self):
        model = Sequential([
            layers.Conv2D(5, 3, input_shape=(10, 10, 5), padding='same')
        ])
        common_test_basis(model, False)

    def testLargeNative(self):
        model = Sequential([
            layers.Conv2D(10, 4, input_shape=(100, 100, 5), padding='same')
        ])
        common_test_basis(model, False)

    def testHugeNative(self):
        raise AssertionError("Causes a segfault in C")
        # model = Sequential([
        #     layers.Conv2D(64, 5, input_shape=(500, 500, 64), padding='same')
        # ])
        # common_test_basis(model, False)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testSmallEnclave(self):
        model = Sequential([
            layers.Conv2D(3, 3, input_shape=(5, 5, 3), padding='same')
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testMediumEnclave(self):
        model = Sequential([
            layers.Conv2D(5, 3, input_shape=(10, 10, 5), padding='same')
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testLargeEnclave(self):
        model = Sequential([
            layers.Conv2D(10, 4, input_shape=(100, 100, 5), padding='same')
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testHugeEnclave(self):
        raise AssertionError("Causes a segfault in C")

        # model = Sequential([
        #     layers.Conv2D(64, 5, input_shape=(500, 500, 64), padding='same')
        # ])
        # common_test_basis(model, True)


if __name__ == "__main__":
    unittest.main()

from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers

import os
import unittest

from frontend.python.common import common_test_basis


class DepthwiseConv2dTests(TensorFlowTestCase):
    @staticmethod
    def testSmallNative():
        model = Sequential([
            layers.DepthwiseConv2D(3, padding='same', input_shape=(5, 5, 3))
        ])
        common_test_basis(model, False)

    @staticmethod
    def testMediumNative():
        model = Sequential([
            layers.DepthwiseConv2D(3, padding='same', input_shape=(10, 10, 5))
        ])
        common_test_basis(model, False)

    @staticmethod
    def testLargeNative():
        model = Sequential([
            layers.DepthwiseConv2D(
                5, padding='same', input_shape=(100, 100, 5))
        ])
        common_test_basis(model, False)

    @unittest.skip
    def testHugeNative(self):
        raise AssertionError("Causes a segfault in C")
        # model = Sequential([
        #     layers.DepthwiseConv2D(
        #         10, padding='same', input_shape=(500, 500, 64))
        # ])
        # common_test_basis(model, False)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testSmallEnclave(self):
        model = Sequential([
            layers.DepthwiseConv2D(3, padding='same', input_shape=(5, 5, 3))
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testMediumEnclave(self):
        model = Sequential([
            layers.DepthwiseConv2D(3, padding='same', input_shape=(10, 10, 5))
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testLargeEnclave(self):
        model = Sequential([
            layers.DepthwiseConv2D(
                5, padding='same', input_shape=(100, 100, 5))
        ])
        common_test_basis(model, True)

    @unittest.skip
    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testHugeEnclave(self):
        raise AssertionError("Causes a segfault in C")
        # model = Sequential([
        #     layers.DepthwiseConv2D(
        #         10, padding='same', input_shape=(500, 500, 64))
        # ])
        # common_test_basis(model, True)


if __name__ == "__main__":
    unittest.main()

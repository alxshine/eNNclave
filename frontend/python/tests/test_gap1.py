from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers

import os
import unittest

from common import common_test_basis


class GlobalAveragePooling1D(TensorFlowTestCase):
    def testSmallNative(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(5, 3))
        ])
        common_test_basis(model, False)

    def testMediumNative(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(10, 5))
        ])
        common_test_basis(model, False)

    def testLargeNative(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(500, 5))
        ])
        common_test_basis(model, False)

    def testHugeNative(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(1000, 64))
        ])
        common_test_basis(model, False)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testSmallEnclave(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(5, 3))
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testMediumEnclave(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(10, 5))
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testLargeEnclave(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(500, 5))
        ])
        common_test_basis(model, True)

    @unittest.skipIf(os.environ.get('SGX_SDK') is None, "SGX is not available")
    def testHugeEnclave(self):
        model = Sequential([
            layers.GlobalAveragePooling1D(input_shape=(1000, 64))
        ])
        common_test_basis(model, True)

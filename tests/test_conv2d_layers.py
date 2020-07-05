from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import numpy as np

from build_enclave import generate_enclave, compile_enclave
from enclave_model import Enclave
import interop.pymatutil as pymatutil

import unittest


def testConv2dLayer(h, w, channels, filters, kernel_size=3):
    # TODO: seed with current date (for consistent results within a day)
    rng = np.random.default_rng()

    inputs = rng.normal(loc=0., scale=2., size=(h, w, channels)).reshape(
        1, h, w, channels).astype(np.float32)
    size = np.prod(inputs.shape)

    test_model = Sequential()
    test_model.add(layers.Input(shape=(h, w, channels), dtype="float32"))
    test_model.add(layers.Conv2D(filters, activation='linear',
                                 kernel_size=kernel_size, padding='same'))

    expected_result = test_model(inputs).numpy().flatten()
    output_size = np.prod(expected_result.shape)

    enclave_model = Enclave(test_model.layers)
    generate_enclave(enclave_model)
    compile_enclave()

    pymatutil.initialize()
    enclave_bytes = pymatutil.enclave_forward(
        inputs.tobytes(), size, output_size)
    enclave_result = np.frombuffer(enclave_bytes, dtype=np.float32)
    np.testing.assert_almost_equal(enclave_result, expected_result,
                                   err_msg="Enclave output is not the same as TensorFlow output")
    pymatutil.teardown()


class Conv2dTests(unittest.TestCase):
    def testSmall(self):
        testConv2dLayer(5, 5, 3, 3)

    def testMedium(self):
        testConv2dLayer(10, 10, 5, 5)

    def testLarge(self):
        testConv2dLayer(100, 100, 5, 10)

    # def testHuge(self):
    #     testConv2dLayer(500, 500, 64, 64) # TODO: find out why this segfaults

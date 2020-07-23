from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import numpy as np

from build_enclave import generate_enclave, compile_enclave
from enclave_model import Enclave
import interop.pymatutil as pymatutil

import unittest


def common(steps, channels, pool_size):
    # TODO: seed with current date (for consistent results within a day)
    rng = np.random.default_rng()

    inputs = rng.normal(loc=0., scale=2., size=(steps, channels)).reshape(
        1, steps, channels).astype(np.float32)
    size = np.prod(inputs.shape)

    test_model = Sequential()
    test_model.add(layers.Input(shape=(steps, channels), dtype="float32"))
    test_model.add(layers.MaxPool2D(pool_size=pool_size))

    expected_result = test_model(inputs).numpy().flatten()
    output_size = np.prod(expected_result.shape)

    enclave_model = Enclave(test_model.layers)
    generate_enclave(enclave_model)
    compile_enclave()

    pymatutil.initialize()
    enclave_bytes = pymatutil.enclave_forward(
        inputs.tobytes(), size, output_size)
    enclave_result = np.frombuffer(enclave_bytes, dtype=np.float32)
    np.testing.assert_almost_equal(enclave_result, expected_result)
    pymatutil.teardown()


class GlobalAveragePooling1D(TensorFlowTestCase):
    def testSmall(self):
        common(5, 3, 3)

    def testMedium(self):
        common(10, 3, 3)

    def testLarge(self):
        common(500, 10, 5)

    def testHuge(self):
        common(1000, 1000, 64)

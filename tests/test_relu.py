from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import numpy as np

from build_enclave import generate_enclave, compile_enclave
from enclave_model import Enclave
import interop.pymatutil as pymatutil

import unittest


def common(size):
    # TODO: seed with current date (for consistent results within a day)
    rng = np.random.default_rng()

    inputs = rng.normal(loc=0., scale=2., size=size).reshape(
        1, size).astype(np.float32)

    test_model = Sequential()
    test_model.add(layers.Input(shape=size, dtype="float32"))
    test_model.add(layers.Dense(size, activation='relu', kernel_initializer='identity'))

    expected_result = test_model(inputs).numpy().flatten()

    enclave_model = Enclave(test_model.layers)
    generate_enclave(enclave_model)
    compile_enclave()

    pymatutil.initialize()
    enclave_bytes = pymatutil.enclave_forward(
        inputs.tobytes(), size, size)
    enclave_result = np.frombuffer(enclave_bytes, dtype=np.float32)
    for n in enclave_result.flatten():
        if n < 0:
            raise AssertionError("ReLu returned negative result")
    np.testing.assert_almost_equal(enclave_result, expected_result,
                                   err_msg="Enclave output is not the same as TensorFlow output")
    pymatutil.teardown()


class ReLuTests(TensorFlowTestCase):
    def testSmall(self):
        common(5)

    def testMedium(self):
        common(10)


    def testLarge(self):
        common(100)

    def testHuge(self):
        common(1000)


if __name__ == "__main__":
    unittest.main()

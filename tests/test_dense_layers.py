from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import numpy as np

from build_enclave import generate_enclave, compile_enclave
from enclave_model import Enclave
import interop.pymatutil as pymatutil

import unittest


def testDenseLayer(size, output_size):
    # TODO: seed with current date (for consistent results within a day)
    rng = np.random.default_rng()

    inputs = rng.normal(loc=0., scale=2., size=size).reshape(
        1, size).astype(np.float32)

    test_model = Sequential()
    test_model.add(layers.Input(shape=size, dtype="float32"))
    test_model.add(layers.Dense(output_size, activation='linear'))

    expected_result = test_model(inputs).numpy().flatten()

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


class DenseTests(unittest.TestCase):
    def testSmall(self):
        testDenseLayer(5, 5)

    def testMedium(self):
        testDenseLayer(10, 10)


    def testLarge(self):
        testDenseLayer(100, 100)

    def testHuge(self):
        testDenseLayer(1000, 1000)


if __name__ == "__main__":
    unittest.main()

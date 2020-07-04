from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers
import numpy as np

from build_enclave import generate_enclave, compile_enclave
from enclave_model import Enclave
import interop.pymatutil as pymatutil

import unittest

class DenseTest(unittest.TestCase):
    def testSmall(self):
        rng = np.random.default_rng()
        size = 5
        output_size = 5
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

if __name__ == "__main__":
    unittest.main()
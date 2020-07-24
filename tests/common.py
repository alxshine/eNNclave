from tensorflow.keras.models import Sequential
import numpy as np


from build_enclave import generate_enclave, compile_enclave
from enclave_model import Enclave
import interop.pymatutil as pymatutil

def common_test_basis(model: Sequential, use_enclave: bool):
    # TODO: seed with current date (for consistent results within a day)
    rng = np.random.default_rng()

    input_shape = model.input_shape[1:]
    inputs = rng.normal(loc=0., scale=2., size=(
        input_shape)).astype(np.float32)
    inputs = np.expand_dims(inputs, 0)
    size = np.prod(inputs.shape)

    expected_result = model(inputs).numpy().flatten()
    output_size = np.prod(expected_result.shape)

    test_model = Enclave(model.layers)
    generate_enclave(test_model)
    compile_enclave()

    if use_enclave:
        pymatutil.initialize()
        test_bytes = pymatutil.enclave_forward(
            inputs.tobytes(), size, output_size)
    else:
        test_bytes = pymatutil.native_forward(
            inputs.tobytes(), size, output_size)

    test_result = np.frombuffer(test_bytes, dtype=np.float32)
    np.testing.assert_almost_equal(test_result, expected_result)

    if use_enclave:
        pymatutil.teardown()
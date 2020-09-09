import numpy as np
import tensorflow.keras.layers as layers
from invoke.context import Context
from tensorflow.keras.models import Sequential

import config as cfg
from ennclave import Enclave
import ennclave_inference


def build_library(model: Enclave, mode: str):
    model.generate_state()
    model.generate_forward(mode, )
    context = Context()
    with context.cd(cfg.get_ennclave_home()):
        if mode == 'sgx':
            model.generate_config()
            context.run('build/backend_sgx_encryptor')

        with context.cd("build"):  # TODO: make more robust
            context.run(f"make backend_{mode}")


def common_test_basis(model: Sequential, use_sgx: bool):
    # TODO: seed with current date (for consistent results within a day)
    rng = np.random.default_rng()

    input_shape = model.input_shape[1:]
    inputs = rng.normal(loc=0., scale=2., size=(
        input_shape)).astype(np.float32)
    inputs = np.expand_dims(inputs, 0)
    size = np.prod(inputs.shape)

    expected_result = model(inputs).numpy().flatten()
    output_size = np.prod(expected_result.shape)

    ennclave_model = Enclave(model.layers)
    build_library(ennclave_model, "sgx" if use_sgx else "native")

    if use_sgx:
        test_bytes = ennclave_inference.sgx_forward(
            inputs.tobytes(), size, output_size)
    else:
        test_bytes = ennclave_inference.native_forward(
            inputs.tobytes(), size, output_size)

    test_result = np.frombuffer(test_bytes, dtype=np.float32)
    np.testing.assert_almost_equal(test_result, expected_result, decimal=5)


if __name__ == '__main__':
    test_shape = (1, 5)

    test_input = np.arange(test_shape[1]).reshape((1,) + test_shape)
    test_model = Sequential(layers.Dense(5, input_shape=test_shape))
    test_model(test_input)
    common_test_basis(test_model, False)
    print("Generated code returns the same result :)")

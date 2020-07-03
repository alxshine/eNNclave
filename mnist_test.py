from tensorflow.keras.models import load_model, Sequential
import numpy as np

from build_enclave import build_enclave
import interop.pymatutil as pymatutil
from mnist_prepare_data import load_test_set

if __name__ == "__main__":
    print("Setting up data")
    x_test, y_test = load_test_set()
    inputs = x_test[:1]
    model_file = 'models/mnist.h5'
    enclave_layers = 1
    model = load_model(model_file)
    enclave_model = build_enclave(model_file, enclave_layers)

    print("\n\nInitializing enclave")
    pymatutil.initialize()

    print("\nBuilding partial models")
    tf_part = Sequential(layers=model.layers[:-enclave_layers])

    print("\nGenerating results")
    expected_result = model(inputs)

    tf_output = tf_part(inputs)
    enclave_input = tf_output.numpy().astype(np.float32)

    rs = np.prod(model.output_shape[1:])
    enclave_result = pymatutil.enclave_forward(enclave_input.tobytes(), np.prod(enclave_input.shape), rs)
    enclave_result_cleaned = np.frombuffer(enclave_result, dtype=np.float32)
    print(enclave_result_cleaned)

    print(expected_result)

    print("\n\nDestroying enclave")
    pymatutil.teardown()
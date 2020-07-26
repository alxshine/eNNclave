from frontend.python.enclave_model import Enclave
import tensorflow.keras.layers as layers
import numpy as np

if __name__ == '__main__':
    model = Enclave([
        layers.Dense(5, input_shape=(5,))
    ])
    inputs = np.arange(5).reshape((1, 5))
    print(f"output: {model(inputs)}")

    model.generate_config()
    model.generate_state()
    model.generate_forward('native')

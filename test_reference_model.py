import h5py
from tensorflow.keras.layers import Dense
from enclave import Enclave
import numpy as np

enclave = Enclave(layers=[
    Dense(128, input_shape=(9216,), activation='relu'),
    Dense(10, activation='softmax')
])

with h5py.File('models/tf_enclave_mnist_cnn.h5', 'r') as f:
    kernel0 = f['model_weights/Enclave/Enclave/dense/kernel:0']
    bias0 = f['model_weights/Enclave/Enclave/dense/bias:0']
    enclave.layers[0].set_weights([kernel0, bias0])
    kernel1 = f['model_weights/Enclave/Enclave/dense_1/kernel:0']
    bias1 = f['model_weights/Enclave/Enclave/dense_1/bias:0']
    enclave.layers[1].set_weights([kernel1, bias1])

arr = np.arange(9216)
arr = arr.reshape((-1, 9216))
print(np.argmax(enclave.predict(arr)))

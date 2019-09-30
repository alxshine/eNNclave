from tensorflow.keras.layers import Dense
from keras_enclave import Enclave

test_model = Enclave(layers=[
    Dense(2, input_shape=(10,), activation='relu'),
    Dense(1, activation='softmax')
    ])
test_model.print_shapes()
test_model.generate_state()
test_model.generate_dense()

from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.utils.layer_utils import print_summary

# # This returns a tensor
# inputs = Input(shape=(28,28,1), name='input')

# # a layer instance is callable on a tensor, and returns a tensor
# conv1 = Conv2D(20, 5, name='conv1')(inputs)
# conv2 = Conv2D(50, 5, name='conv2')(conv1)
# flatten = Flatten(name='flatten')(conv2)
# fc1 = Dense(500, activation='relu',name='fc1')(flatten)
# fc2 = Dense(10, activation='relu',name='fc2')(fc1)
# predictions = Dense(10, activation='softmax',name='softmax')(fc2)

# # This creates a model that includes
# # the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=predictions, name='Sequential API')
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()

dense_part = Sequential([
    Dense(500, activation='relu',name='fc1'),
    Dense(10, activation='relu',name='fc2'),
    Dense(10, activation='softmax',name='softmax')],
    name='SGX')

model = Sequential([
    Conv2D(20,5, name='conv1', input_shape=(28,28,1)),
    Conv2D(50,5, name='conv2'),
    Flatten(name='flatten'),
    dense_part
], name='Model API')
# model.summary()
# plot_model(model, to_file='mnist_cnn.png', expand_nested=True)

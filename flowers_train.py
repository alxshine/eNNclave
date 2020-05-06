import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
import tensorflow.keras.layers as layers

from flowers_prepare_data import load_data
import utils

IMG_SIZE = 224

BATCH_SIZE = 32

x_train, y_train, x_test, y_test = load_data()

extractor = VGG19(include_top=False, input_shape=(224,224,3))

model = Sequential()

all_layers = utils.get_all_layers(extractor)
for l in all_layers:
    l.trainable = False
    model.add(l)

model.add(layers.Flatten())
model.add(layers.Dense(800, activation='relu'))
model.add(layers.Dense(800, activation='relu'))
model.add(layers.Dense(600, activation='relu'))

model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,
        epochs = 20,
	verbose = 0,
        )

loss0, accuracy0 = model.evaluate(x_test, y_test)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

model.save('models/flowers.h5')

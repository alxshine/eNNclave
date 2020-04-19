import pathlib
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNet
import tensorflow.keras.layers as layers

import utils

data_dir = 'datasets/flowers'
data_dir = pathlib.Path(data_dir)

label_names = {'daisy': 0, 'dandelion': 1,
               'rose': 2, 'sunflower': 3, 'tulip': 4}
label_key = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulip']
all_images = list(data_dir.glob('*/*'))
all_images = [str(path) for path in all_images]
random.shuffle(all_images)

all_labels = [label_names[pathlib.Path(path).parent.name]
              for path in all_images]

data_size = len(all_images)

train_test_split = (int)(data_size*0.2)

x_train = all_images[train_test_split:]
x_test = all_images[:train_test_split]

y_train = all_labels[train_test_split:]
y_test = all_labels[:train_test_split]

IMG_SIZE = 224

BATCH_SIZE = 32

train_ds = utils.generate_dataset(x_train, y_train)
validation_ds = utils.generate_dataset(x_test, y_test, repeat=False)

mobilenet = MobileNet(include_top = False, input_shape=(224,224,3))

model = Sequential()

for l in mobilenet.layers:
    if type(l).__name__ == 'BatchNormalization':
        continue
    else:
        l.trainable = False
        model.add(l)

model.add(layers.MaxPooling2D(7))
model.add(layers.Flatten())

# add dense layers
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
history = model.fit(train_ds,
        epochs = 100,
        steps_per_epoch=10,
        )

loss0, accuracy0 = model.evaluate(validation_ds)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

model.save('models/flowers.h5')

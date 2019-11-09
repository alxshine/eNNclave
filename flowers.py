import pathlib
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

import utils

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = 'data/flowers'
data_dir = pathlib.Path(data_dir)

label_names = {'daisy': 0, 'dandelion': 1,
               'roses': 2, 'sunflowers': 3, 'tulips': 4}
label_key = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
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
validation_ds = utils.generate_dataset(x_test, y_test)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model_file = 'models/vgg_flowers.h5'

VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                          include_top=False,
                                          weights='imagenet')
VGG16_MODEL.trainable = False

HIDDEN_NEURONS = 4096

model = tf.keras.Sequential([
    VGG16_MODEL,
    GlobalAveragePooling2D(),
    Dense(HIDDEN_NEURONS, activation='relu'),
    Dense(HIDDEN_NEURONS, activation='relu'),
    Dense(len(label_names), activation='softmax')
])
breakpoint()
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
history = model.fit(train_ds,
                    epochs=100,
                    steps_per_epoch=2,
                    validation_steps=2,
                    validation_data=validation_ds)
model.save(model_file)


validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_ds, steps=validation_steps)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

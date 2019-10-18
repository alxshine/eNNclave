import tensorflow as tf
import numpy as np
import os

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = 'data/flowers/processed'
x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 32
DATA_SIZE = x_train.shape[0] + x_test.shape[0]
train_dataset = train_dataset.shuffle(buffer_size=DATA_SIZE)
train_dataset = train_dataset.repeat().batch(
    BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.shuffle(buffer_size=DATA_SIZE)
test_dataset = test_dataset.repeat().batch(
    BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
HIDDEN_NEURONS = 4096

model_file = 'models/vgg_flowers_new.h5'

VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                          include_top=False,
                                          weights='imagenet')
VGG16_MODEL.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(
    5, activation='softmax')
model = tf.keras.Sequential([
    VGG16_MODEL,
    global_average_layer,
    prediction_layer
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
history = model.fit(train_dataset,
                    epochs=100,
                    steps_per_epoch=2,
                    validation_steps=2,
                    validation_data=test_dataset)
model.save(model_file)

# response = input("Would you like to measure prediction time? [y/N]")
# if response == 'y':
#     validation_steps = 1
#     time_before = time.process_time()
#     model.predict_generator(validation_ds, steps=validation_steps)
#     time_after = time.process_time()

#     print("Prediction on %d samples took %s seconds" %
#           (validation_steps*BATCH_SIZE, time_after - time_before))

import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy

import pandas as pd

img_rows, img_cols = 28, 28
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# build model
MODEL_FILE = 'models/mnist.h5'
HIST_FILE = 'hist_mnist.csv'
HIDDEN_NEURONS = 1024
DROPOUT_RATIO=0.4
NUM_EPOCHS = 1
STEPS_PER_EPOCH = 10

model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(DROPOUT_RATIO),
    Dense(128, activation='relu'),
    Dropout(DROPOUT_RATIO),
    Dense(num_classes, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss=sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=(x_test, y_test))

loss0, accuracy0 = model.evaluate(x_test, y_test)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(MODEL_FILE))
model.save(MODEL_FILE)

print("Saving history at: {}".format(HIST_FILE))
hist_df = pd.DataFrame(history.history)
with open(HIST_FILE, 'w+') as f:
    hist_df.to_csv(f)

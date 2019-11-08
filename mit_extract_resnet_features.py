from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import numpy as np

from os.path import join
import os


def preprocess(x):
    img_size = 224
    image = tf.io.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (img_size, img_size))
    return image


# global config
data_dir = 'data/mit67'
train_file = 'TrainImages.txt'
test_file = 'TestImages.txt'

# build label mapping from directories
labels = {}
i = 0
for entry in os.scandir(data_dir):
    if entry.is_dir():
        labels[entry.name] = i
        i += 1

# load images
with open(join(data_dir, train_file), 'r') as f:
    train_images = [join(data_dir, l.strip()) for l in f]
with open(join(data_dir, test_file), 'r') as f:
    test_images = [join(data_dir, l.strip()) for l in f.readlines()]

# generate labels
train_labels = [labels[s.split('/')[-2]] for s in train_images]
test_labels = [labels[s.split('/')[-2]] for s in test_images]


# generate datasets
print('Generating datasets')
x_train = list(map(preprocess, train_images))
x_train = [tf.expand_dims(t, axis=0)
           for t in x_train]  # get tensors to right dimensionality
y_train = np.array(train_labels)

x_test = list(map(preprocess, test_images))
x_test = [tf.expand_dims(t, axis=0) for t in x_test]
y_test = np.array(test_labels)

resnet = ResNet50(include_top=False, weights='imagenet',
                  input_shape=((224, 224, 3)))

print('Extracting features')
train_features = resnet.predict(x_train, steps=len(x_train))
test_features = resnet.predict(x_test, steps=len(x_test))

print('Done, saving results')
np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_test', x_test)
np.save('y_test', y_test)

import pathlib
import random
import os

import numpy as np
import tensorflow as tf

import utils

def load_data():
    random.seed(1337)

    data_dir = 'datasets/flowers'
    try:
        print("Trying to load previously generated data")
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
    except IOError:
        print("Not found, generating...")
        data_files = pathlib.Path(data_dir)

        label_names = {'daisy': 0, 'dandelion': 1,
                       'rose': 2, 'sunflower': 3, 'tulip': 4}
        label_key = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulip']
        all_images = list(data_files.glob('*/*'))
        all_images = [str(path) for path in all_images]
        random.shuffle(all_images)

        all_labels = [label_names[pathlib.Path(path).parent.name]
                      for path in all_images]

        data_size = len(all_images)

        train_test_split = (int)(data_size*0.8)

        train_images = all_images[:train_test_split]
        x_train = np.empty((len(train_images), 224, 224, 3))
        for i, image in enumerate(train_images):
            image = tf.io.read_file(image)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            image = (image/127.5) - 1
            image = tf.image.resize(image, (224, 224))
            x_train[i] = image

        test_images = all_images[train_test_split:]
        x_test = np.empty((len(test_images), 224, 224, 3))
        for i, image in enumerate(test_images):
            image = tf.io.read_file(image)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            image = (image/127.5) - 1
            image = tf.image.resize(image, (224, 224))
            x_test[i] = image

        y_train = np.array(all_labels[:train_test_split])
        y_test = np.array(all_labels[train_test_split:])

        print("Saving numpy arrays")
        np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
        np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
        np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(data_dir, 'y_train.npy'), y_train)

    return x_train, y_train, x_test, y_test

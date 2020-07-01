from tensorflow.keras.datasets import mnist
import numpy as np
from os.path import join, isdir
import os


def load_test_set(data_dir='datasets'):
    try:
        x_test = np.load(join(data_dir, "mnist/x_test.npy"))
        y_test = np.load(join(data_dir, "mnist/y_test.npy"))
    except IOError:
        os.makedirs(join(data_dir, 'mnist'), exist_ok=True)

        _, (x_test, y_test) = mnist.load_data()
        x_test = (x_test/255).reshape(-1, 28, 28, 1)
        np.save(join(data_dir, "mnist/x_test.npy"), x_test)
        np.save(join(data_dir, "mnist/y_test.npy"), y_test)

    return x_test, y_test


def load_train_set(data_dir='datasets'):
    try:
        x_train = np.load(join(data_dir, "mnist/x_train.npy"))
        y_train = np.load(join(data_dir, "mnist/y_train.npy"))
    except IOError:
        os.makedirs(join(data_dir, 'mnist'), exist_ok=True)

        (x_train, y_train), _ = mnist.load_data()
        x_train = (x_train/255).reshape(-1, 28, 28, 1)
        np.save(join(data_dir, "mnist/x_train.npy"), x_train)
        np.save(join(data_dir, "mnist/y_train.npy"), y_train)

    return x_train, y_train

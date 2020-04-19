""" Taken from GoogleML training git at https://github.com/google/eng-edu """

"""Module to train n-gram model.

Vectorizes training and validation texts into n-grams and uses that for
training a n-gram model - a simple multi-layer perceptron model. We use n-gram
model for text classification when the ratio of number of samples to number of
words per sample for the given dataset is very small (<~1500).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import tensorflow as tf
import numpy as np

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

from imdb_prepare_data import load_imdb
import utils

FLAGS = None

def mlp_model(num_layers, units, dropout_rate, input_shape):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        num_layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.

    # Returns
        An MLP model instance.
    """
    model = models.Sequential()
    model.add(layers.Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(num_layers-1):
        model.add(layers.Dense(units=units, activation='relu'))
        model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(units=1, activation='sigmoid'))
    return model

def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      num_layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        num_layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    x_train, y_train, x_test, y_test = data

    # Create model instance.
    model = mlp_model(num_layers=num_layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:])

    # Compile model with learning parameters.
    loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)]

    # Train and validate model.
    history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_test, y_test),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('models/imdb.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    # Using the IMDb movie reviews dataset to demonstrate training n-gram model
    data = load_imdb('./datasets')
    train_ngram_model(data)

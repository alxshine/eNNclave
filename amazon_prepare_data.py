import os
import random

import numpy as np
import pandas as pd
import json

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import utils

def load_amazon(data_path):
    amazon_path = os.path.join(data_path, 'amazon')

    try:
        # try and load saved np arrays
        print("Trying to load previously generated data and labels")
        x_train = np.load(os.path.join(amazon_path, 'x_train.npy'))
        y_train = np.load(os.path.join(amazon_path, 'y_train.npy'))
        x_test = np.load(os.path.join(amazon_path, 'x_test.npy'))
        y_test = np.load(os.path.join(amazon_path, 'y_test.npy'))

        with open(os.path.join(amazon_path, 'word_index.json'), 'r') as f:
            word_index = json.load(f)
            
    except IOError:
        # generate numpy arrays for future use
        print("No data found, generating...")
            
        validation_split = 0.2
        seed = 1337
        data = load_amazon_sentiment_analysis_dataset(amazon_path,
                                                               validation_split,
                                                               seed)
        (train_texts, y_train), (val_texts, y_test) = data

        # Verify that validation labels are in the same range as training labels.
        num_classes = utils.get_num_classes(y_train)
        unexpected_labels = [v for v in y_test if v not in range(num_classes)]
        if len(unexpected_labels):
            raise ValueError('Unexpected label values found in the validation set:'
                             ' {unexpected_labels}. Please make sure that the '
                             'labels in the validation set are in the same range '
                             'as training labels.'.format(
                                 unexpected_labels=unexpected_labels))

        # Vectorize texts.
        x_train, x_test, word_index = utils.sequence_vectorize(
                train_texts, val_texts)
        
        np.save(os.path.join(amazon_path, 'x_train.npy'), x_train)
        np.save(os.path.join(amazon_path, 'y_train.npy'), y_train)
        np.save(os.path.join(amazon_path, 'x_test.npy'), x_test)
        np.save(os.path.join(amazon_path, 'y_test.npy'), y_test)
        with open(os.path.join(amazon_path, 'word_index.json'), 'w+') as f:
            json.dump(word_index, f)

    return x_train, y_train, x_test, y_test, word_index

def load_amazon_sentiment_analysis_dataset(amazon_path,
                                                    validation_split,
                                                    seed):
    columns = (1, 0)  # 2 - Phrases, 1 - Sentiment.
    data = utils.load_and_shuffle_data(amazon_path, 'train_cleaned.txt', columns, seed, '\t')

    # Get the review phrase and sentiment values.
    texts = list(data['Phrase'])
    labels = np.array(data['Sentiment'])
    return utils.split_training_and_validation_sets(texts, labels, validation_split)

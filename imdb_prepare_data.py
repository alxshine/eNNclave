""" Based off of GoogleML training git at https://github.com/google/eng-edu """
import os
import random

import numpy as np
import pandas as pd

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Vectorization parameters

# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500

def load_imdb(data_path):
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    try:
        # try and load saved np arrays
        print("Trying to load previously generated data and labels")
        x_train = np.load(os.path.join(imdb_data_path, 'x_train.npy'))
        y_train = np.load(os.path.join(imdb_data_path, 'y_train.npy'))
        x_test = np.load(os.path.join(imdb_data_path, 'x_test.npy'))
        y_test = np.load(os.path.join(imdb_data_path, 'y_test.npy'))
    except IOError:
        # generate numpy arrays for future use
        print("No data found, generating...")
        data = load_imdb_sentiment_analysis_dataset(imdb_data_path)
        (train_texts, train_labels), (val_texts, val_labels) = data

        # Verify that validation labels are binary
        unexpected_labels = [v for v in val_labels if v not in range(2)]
        # Same thing for training data
        unexpected_labels += [v for v in val_labels if v not in range(2)]
        if len(unexpected_labels):
            raise ValueError('Unexpected label values found in the validation set:'
                             ' {unexpected_labels}. Please make sure that the '
                             'labels in the training and validation set are binary')

        # Vectorize texts.
        x_train, x_test = ngram_vectorize(
            train_texts, train_labels, val_texts)
        y_train = train_labels
        y_test = val_labels

        np.save(os.path.join(imdb_data_path, 'x_train.npy'), x_train)
        np.save(os.path.join(imdb_data_path, 'y_train.npy'), y_train)
        np.save(os.path.join(imdb_data_path, 'x_test.npy'), x_test)
        np.save(os.path.join(imdb_data_path, 'y_test.npy'), y_test)

    return x_train, y_train, x_test, y_test


def load_imdb_sentiment_analysis_dataset(imdb_data_path, seed=123):
    """Loads the Imdb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as ngram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': np.float32,
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)

    x_train = x_train.astype('float32').toarray()
    x_val = x_val.astype('float32').toarray()
    return x_train, x_val

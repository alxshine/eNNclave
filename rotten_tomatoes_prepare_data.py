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

def load_rotten_tomatoes(data_path):
    rotten_tomatoes_path = os.path.join(data_path, 'rotten_tomatoes')

    try:
        # try and load saved np arrays
        print("Trying to load previously generated data and labels")
        x_train = np.load(os.path.join(rotten_tomatoes_path, 'x_train.npy'))
        y_train = np.load(os.path.join(rotten_tomatoes_path, 'y_train.npy'))
        x_test = np.load(os.path.join(rotten_tomatoes_path, 'x_test.npy'))
        y_test = np.load(os.path.join(rotten_tomatoes_path, 'y_test.npy'))

        with open(os.path.join(rotten_tomatoes_path, 'word_index.json'), 'r') as f:
            word_index = json.load(f)
            
    except IOError:
        # generate numpy arrays for future use
        print("No data found, generating...")
            
        validation_split = 0.2
        seed = 1337
        data = load_rotten_tomatoes_sentiment_analysis_dataset(rotten_tomatoes_path,
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
        x_train, x_test, word_index = sequence_vectorize(
                train_texts, val_texts)
        
        np.save(os.path.join(rotten_tomatoes_path, 'x_train.npy'), x_train)
        np.save(os.path.join(rotten_tomatoes_path, 'y_train.npy'), y_train)
        np.save(os.path.join(rotten_tomatoes_path, 'x_test.npy'), x_test)
        np.save(os.path.join(rotten_tomatoes_path, 'y_test.npy'), y_test)
        with open(os.path.join(rotten_tomatoes_path, 'word_index.json'), 'w+') as f:
            json.dump(word_index, f)

    return x_train, y_train, x_test, y_test, word_index

def load_rotten_tomatoes_sentiment_analysis_dataset(rotten_tomatoes_path,
                                                    validation_split,
                                                    seed):
    """Loads the rotten tomatoes sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data base directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 124848
        Number of test samples: 31212
        Number of categories: 5 (0 - negative, 1 - somewhat negative,
                2 - neutral, 3 - somewhat positive, 4 - positive)

    # References
        https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

        Download and uncompress archive from:
        https://www.kaggle.com/c/3810/download/train.tsv.zip
    """
    columns = (2, 3)  # 2 - Phrases, 3 - Sentiment.
    data = _load_and_shuffle_data(rotten_tomatoes_path, 'train.tsv', columns, seed, '\t')

    # Get the review phrase and sentiment values.
    texts = list(data['Phrase'])
    labels = np.array(data['Sentiment'])
    return _split_training_and_validation_sets(texts, labels, validation_split)

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

def _load_and_shuffle_data(data_path,
                           file_name,
                           cols,
                           seed,
                           separator=',',
                           header=0):
    """Loads and shuffles the dataset using pandas.

    # Arguments
        data_path: string, path to the data directory.
        file_name: string, name of the data file.
        cols: list, columns to load from the data file.
        seed: int, seed for randomizer.
        separator: string, separator to use for splitting data.
        header: int, row to use as data header.
    """
    np.random.seed(seed)
    data_path = os.path.join(data_path, file_name)
    data = pd.read_csv(data_path, usecols=cols, sep=separator, header=header)
    return data.reindex(np.random.permutation(data.index))


def _split_training_and_validation_sets(texts, labels, validation_split):
    """Splits the texts and labels into training and validation sets.

    # Arguments
        texts: list, text data.
        labels: list, label data.
        validation_split: float, percentage of data to use for validation.

    # Returns
        A tuple of training and validation data.
    """
    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))

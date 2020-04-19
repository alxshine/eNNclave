import tensorflow.keras.preprocessing.text as text
import tensorflow as tf
import tensorflow.keras.preprocessing.sequence as sequence

import pandas as pd
import numpy as np

from os.path import join
import json

TRAIN_SPLIT = 0.8
DATA_DIR = 'datasets/amazon'

TOKENIZER_CONFIG_FILE = 'datasets/amazon/tokenizer_config.json'

def load_books(num_words, sequence_length, seed = 1337, verbose = 0):
    print('Loading books')
    data_dir = join(DATA_DIR, 'books')
    data = pd.read_pickle(join(data_dir, 'books.pkl'))

    train_data = pd.DataFrame(columns=data.columns)
    test_data = pd.DataFrame(columns=data.columns)

    for i in range(1,6):
        current_data = data.where(data['rating'] == i)
        current_train_data = current_data.sample(frac=TRAIN_SPLIT, replace=False, random_state=seed).dropna(how='all')
        current_test_data = current_data[~current_data.isin(current_train_data)].dropna(how='all')

        train_data = train_data.append(current_train_data)
        test_data = test_data.append(current_test_data)

    train_data = train_data.sample(frac=1, replace=False, random_state=seed)
    test_data = test_data.sample(frac=1, replace=False, random_state=seed)

    if verbose:
        print("train data value counts:")
        print(train_data['rating'].value_counts())
        print("test data value counts:")
        print(test_data['rating'].value_counts())

    train_texts = train_data['text']
    y_train = np.array(train_data['rating'])

    test_texts = test_data['text']
    y_test = np.array(test_data['rating'])

    try:
        with open(TOKENIZER_CONFIG_FILE, 'r') as f:
            print("Tokenizer config found, loading...")
            tokenizer = text.tokenizer_from_json(f.read())
    except IOError:
        print("Generating tokenizer...")
        tokenizer = text.Tokenizer(num_words)
        tokenizer.fit_on_texts(train_texts)

        with open(TOKENIZER_CONFIG_FILE, 'w+') as f:
            print("Saving tokenizer...")
            json_dict = {'config': tokenizer.get_config()}
            json.dump(json_dict, f)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    x_train = sequence.pad_sequences(train_sequences, maxlen=sequence_length)
    x_test = sequence.pad_sequences(test_sequences, maxlen=sequence_length)

    return x_train, y_train, x_test, y_test

def load_cds(num_words, sequence_length, seed = 1337, verbose = 0):
    print('Loading cds')
    data_dir = join(DATA_DIR, 'cds')
    data = pd.read_pickle(join(data_dir, 'cds.pkl'))

    train_data = pd.DataFrame(columns=data.columns)
    test_data = pd.DataFrame(columns=data.columns)

    for i in range(1,6):
        current_data = data.where(data['rating'] == i)
        current_train_data = current_data.sample(frac=TRAIN_SPLIT, replace=False, random_state=seed).dropna(how='all')
        current_test_data = current_data[~current_data.isin(current_train_data)].dropna(how='all')

        train_data = train_data.append(current_train_data)
        test_data = test_data.append(current_test_data)

    train_data = train_data.sample(frac=1, replace=False, random_state=seed)
    test_data = test_data.sample(frac=1, replace=False, random_state=seed)

    if verbose:
        print("train data value counts:")
        print(train_data['rating'].value_counts())
        print("test data value counts:")
        print(test_data['rating'].value_counts())

    train_texts = train_data['text']
    y_train = np.array(train_data['rating'])

    test_texts = test_data['text']
    y_test = np.array(test_data['rating'])

    try:
        with open(TOKENIZER_CONFIG_FILE, 'r') as f:
            print("Tokenizer config found, loading...")
            tokenizer = text.tokenizer_from_json(f.read())
    except IOError:
        print("Generating tokenizer...")
        tokenizer = text.Tokenizer(num_words)
        tokenizer.fit_on_texts(train_texts)

        with open(TOKENIZER_CONFIG_FILE, 'w+') as f:
            print("Saving tokenizer...")
            json_dict = {'config': tokenizer.get_config()}
            json.dump(json_dict, f)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    x_train = sequence.pad_sequences(train_sequences, maxlen=sequence_length)
    x_test = sequence.pad_sequences(test_sequences, maxlen=sequence_length)

    return x_train, y_train, x_test, y_test

def rebuild_cds(num_words, sequence_length, seed = 1337):
    print('Loading cds')
    data_dir = join(DATA_DIR, 'cds')
    data = pd.read_pickle(join(data_dir, 'cds.pkl'))

    train_data = pd.DataFrame(columns=data.columns)
    test_data = pd.DataFrame(columns=data.columns)

    for i in range(1,6):
        current_data = data.where(data['rating'] == i)
        current_train_data = current_data.sample(frac=TRAIN_SPLIT, replace=False, random_state=seed).dropna(how='all')
        current_test_data = current_data[~current_data.isin(current_train_data)].dropna(how='all')

        train_data = train_data.append(current_train_data)
        test_data = test_data.append(current_test_data)

    train_data = train_data.sample(frac=1, replace=False, random_state=seed)
    test_data = test_data.sample(frac=1, replace=False, random_state=seed)

    print("train data value counts:")
    print(train_data['rating'].value_counts())
    print("test data value counts:")
    print(test_data['rating'].value_counts())

    train_texts = train_data['text']
    y_train = np.array(train_data['rating'])

    test_texts = test_data['text']
    y_test = np.array(test_data['rating'])

    print("Generating tokenizer...")
    tokenizer = text.Tokenizer(num_words)
    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    x_train = sequence.pad_sequences(train_sequences, maxlen=sequence_length)
    x_test = sequence.pad_sequences(test_sequences, maxlen=sequence_length)

    return x_train, y_train, x_test, y_test, tokenizer

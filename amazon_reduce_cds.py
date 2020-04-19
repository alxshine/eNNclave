import tensorflow.keras.preprocessing.text as pre_text
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers

import numpy as np
import pandas as pd

import json
import os
import plotille

SEED = 42

DATA_DIR = 'datasets/amazon/cds'
JSON_FILE = 'CDs_and_Vinyl.json'
PICKLE_FILE = 'cds.pkl'

SAMPLES_PER_CLASS = 200
#  SAMPLES_PER_CLASS = 1850565
TOTAL_ROWS = 51311621

sample_counts = {1:0, 2:0, 3:0, 4:0, 5:0}

def _check_complete():
    for v in sample_counts.values():
        if v < SAMPLES_PER_CLASS:
            return False
    return True

ratings = []
texts = []

with open(os.path.join(DATA_DIR, JSON_FILE), 'r') as input_file:
    for line in input_file:
        json_dict = json.loads(line)
        try:
            text = json_dict['reviewText']
            rating = int(json_dict['overall'])

            if sample_counts[rating] >= SAMPLES_PER_CLASS:
                continue

            ratings.append(rating)
            texts.append(text)
            sample_counts[rating] += 1
        except KeyError:
            continue

        if _check_complete():
            break

data = pd.DataFrame()
data['rating'] = pd.Series(ratings)
data['text'] = pd.Series(texts, index=data.index)
data.to_pickle(os.path.join(DATA_DIR, PICKLE_FILE))

print("Sample counts:")
print(sample_counts)

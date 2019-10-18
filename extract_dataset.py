# coding: utf-8
import numpy as np
import os
from PIL import Image


def extract_from_dir(dir_path):
    print('Generating samples from %s' % entry.name)
    contained_files = next(os.walk(dir_path))[2]
    samples = np.empty((len(contained_files),) + input_shape)

    for i, f in enumerate(contained_files):
        full_path = os.path.join(dir_path, f)
        img = Image.open(full_path)
        resized = img.resize(input_shape[:2])
        samples[i] = np.array(resized)

    target_path = os.path.join(PROCESSED_DIR, entry.name)
    np.save(target_path, samples)


def filter_func(x):
    if x in ['x_test.npy', 'y_test.npy', 'x_train.npy', 'y_train.npy']:
        return False
    return x.endswith('.npy')


DATASET_DIR = 'data/flowers'
RAW_DIR = os.path.join(DATASET_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATASET_DIR, 'processed')
TRAINING_RATIO = 0.8

if not os.path.exists(PROCESSED_DIR):
    os.mkdir(PROCESSED_DIR)


response = input('Would you like to generate numpy arrays from images?(y/N)')
if response == 'y':
    input_shape = ((224, 224, 3))
    for entry in os.scandir(RAW_DIR):
        if not entry.is_dir():
            continue

        extract_from_dir(entry.path)

all_files = next(os.walk(PROCESSED_DIR))[2]


response = input('Generate training and test sets?(y/N)')
if response == 'y':
    numpy_files = list(filter(filter_func, all_files))
    numpy_arrs = [np.load(os.path.join(PROCESSED_DIR, f)) for f in numpy_files]

    label_counts = [n.shape[0] for n in numpy_arrs]
    total_samples = sum(label_counts)
    input_shape = numpy_arrs[0].shape[1:]
    all_samples = np.empty((total_samples,) + input_shape)
    all_labels = np.zeros((total_samples, len(numpy_files)))

    start = 0
    end = 0

    # build samples and labels
    print('Combining all datasets')
    label = 0
    for a in numpy_arrs:
        end += a.shape[0]
        all_samples[start:end] = a
        all_labels[start:end, label] = 1
        start += a.shape[0]
        label += 1

    # normalize if needed
    if all_samples.max() > 1:
        print('Normalizing values')
        all_samples /= 255

    # shuffle samples and labels
    print('Shuffling and performing training/test split')
    shuffled_indices = np.random.permutation(total_samples)
    split_index = int(total_samples*TRAINING_RATIO)

    x_train = all_samples[shuffled_indices][:split_index]
    np.save(os.path.join(PROCESSED_DIR, 'x_train'), x_train)
    y_train = all_labels[shuffled_indices][:split_index]
    np.save(os.path.join(PROCESSED_DIR, 'y_train'), y_train)

    x_test = all_samples[shuffled_indices][split_index:]
    np.save(os.path.join(PROCESSED_DIR, 'x_test'), x_test)
    y_test = all_labels[shuffled_indices][split_index:]
    np.save(os.path.join(PROCESSED_DIR, 'y_test'), y_test)

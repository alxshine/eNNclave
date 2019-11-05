import pathlib
import os
import json
import numpy as np


def build_lfw(num_classes):
    data_dir = 'data/lfw'
    data_dir = pathlib.Path(data_dir)
    label_file_name = os.path.join(data_dir, 'labelings.json')

    with open(label_file_name, 'r') as label_file:
        labels = json.load(label_file)

    all_paths = list(data_dir.glob('*/*'))
    all_images = [str(path) for path in all_paths]
    all_labels = np.array([labels[path.parent.name] for path in all_paths])

    # sort labels by their count
    uniques, counts = np.unique(all_labels, return_counts=True)
    # zip returns a generator, so it's only good for one use
    zipped = zip(uniques, counts)
    sorted_tuples = sorted(zipped, key=lambda x: x[1], reverse=True)

    # select subset of classes, ordered by number of samples
    included_classes, _ = set(zip(*sorted_tuples[:num_classes]))
    breakpoint()
    x = list(
        map(lambda t: t[1],
            filter(lambda t: all_labels[t[0]] in included_classes,
                   enumerate(all_images))))
    y = list(
        filter(lambda l: l in included_classes,
               all_labels))

    # reformat labels to [0,num_classes)
    new_class_mapping = {}
    for i, c in enumerate(included_classes):
        new_class_mapping[c] = i
    y = list(
        map(lambda l: new_class_mapping[l], y))

    return x, y

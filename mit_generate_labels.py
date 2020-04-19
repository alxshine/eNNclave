from os.path import join
import os

import json

label_file = "datasets/mit67/TestImages.txt"

with open(label_file, 'r') as f:
    all_labels = [l.split('/')[0] for l in f]

labels = set(all_labels)
label_dict = {}
i = 0
for l in labels:
    label_dict[l] = i
    i += 1

target_file = "datasets/mit67/class_labels.json"
with open(target_file, 'w+') as f:
    json.dump(label_dict, f, indent=2)

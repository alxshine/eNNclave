# coding: utf-8
import pandas as pd
import numpy as np

csv = pd.read_csv('identity_CelebA.txt', sep=' ', header=None)
labels = np.array(csv[1])
np.save('labels.npy', labels[:10000])

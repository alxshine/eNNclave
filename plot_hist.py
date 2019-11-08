# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt

hist = pd.read_csv('hist_mit_vgg.csv')
indices = range(len(hist))
plt.plot(indices, hist.acc, label='accuracy')
plt.plot(indices, hist.val_acc, label='validation accuracy')
plt.legend()
plt.show()

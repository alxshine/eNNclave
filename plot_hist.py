# coding: utf-8
import pandas as pd
import plotille as plt

import sys

if len(sys.argv) < 2:
    print("Usage {} hist_file".format(sys.argv[0]))
    sys.exit(1)

hist = pd.read_csv(sys.argv[1])
indices = range(len(hist))
try:
    acc = hist.acc
except AttributeError:
    acc = hist.accuracy

try:
    val_acc = hist.val_acc
except AttributeError:
    val_acc = hist.val_accuracy

fig = plt.Figure()
fig.set_x_limits(min_=0, max_=len(hist))
fig.plot(indices, acc, label='accuracy')
fig.plot(indices, val_acc, label='validation accuracy')
print(fig.show(legend=True))

print('Final training accuracy: {}'.format(acc.iloc[-1]))
print('Final validation accuracy: {}'.format(val_acc.iloc[-1]))
print()
print('Max training accuracy: {}'.format(acc.max()))
print('Max validation accuracy: {}'.format(val_acc.max()))

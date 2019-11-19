# coding: utf-8
import pandas as pd
import plotille as plt

import sys

if len(sys.argv) < 2:
    print("Usage {} hist_file".format(sys.argv[0]))

hist = pd.read_csv(sys.argv[1])
indices = range(len(hist))
fig = plt.Figure()
fig.set_x_limits(min_=0, max_=len(hist))
fig.plot(indices, hist.acc, label='accuracy')
fig.plot(indices, hist.val_acc, label='validation accuracy')
print(fig.show(legend=True))

last_row = hist.iloc[-1]
acc = last_row['acc']
val_acc = last_row['val_acc']
print('Final training accuracy: {}'.format(acc))
print('Final validation accuracy: {}'.format(val_acc))
print()
print('Max training accuracy: {}'.format(hist.acc.max()))
print('Max validation accuracy: {}'.format(hist.val_acc.max()))

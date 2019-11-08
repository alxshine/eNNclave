# coding: utf-8
import numpy as np
x_train = np.load('x_train.npy')
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = np.load('x_test.npy')
x_test = x_test.reshape((x_test.shape[0], -1))
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier()
svm.fit(x_train,y_train)
get_ipython().run_line_magic('save', 'train_svm.py ~0/')

# coding: utf-8
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import numpy as np

print('Loading data')
x_train = np.load('x_train.npy')
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = np.load('x_test.npy')
x_test = x_test.reshape((x_test.shape[0], -1))
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

x_train = x_train[:10]
y_train = y_train[:10]

print('Training classifier')
svm = SGDClassifier()
svm.fit(x_train, y_train)

model_file = 'svm.joblib'
print('Saving model to {}'.format(model_file))
dump(svm, model_file)

predictions = svm.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print("Testing accuracy: {}".format(accuracy))

from tensorflow.keras.models import load_model

import numpy as np

x_train = np.load('datasets/amazon/x_train.npy')
y_train = np.load('datasets/amazon/y_train.npy')
x_test = np.load('datasets/amazon/x_test.npy')
y_test = np.load('datasets/amazon/y_test.npy')

model = load_model('models/amazon.h5')

print("Generating true training accuracy")
train_predictions = model.predict(x_train, verbose = 0)
train_cleaned_predictions = train_predictions.flatten().round()
train_acc = np.mean(train_cleaned_predictions == y_train)

print("Generating true test accuracy")
test_predictions = model.predict(x_test, verbose = 0)
test_cleaned_predictions = test_predictions.flatten().round()
test_acc = np.mean(test_cleaned_predictions == y_test)

print(f'True training accuracy: {train_acc*100:.4}')
print(f'True validation accuracy: {test_acc*100:.4}')

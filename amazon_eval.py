from tensorflow.keras.models import load_model

import numpy as np

x_test = np.load('datasets/amazon/x_test.npy')
y_test = np.load('datasets/amazon/y_test.npy')

model = load_model('models/amazon.h5')

print("Predicting on test data...")
predictions = model.predict(x_test, verbose=0)
print("DONE!")

cleaned_predictions = predictions.flatten().round()
acc = np.mean(cleaned_predictions == y_test)

print(f"Total validation accuracy: {acc:.2}")

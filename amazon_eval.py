from tensorflow.keras.models import load_model

import numpy as np

from amazon_prepare_data import load_books

def eval_true_accuracy(model, x_train, y_train, x_test, y_test)
    print("Generating true training accuracy")
    train_predictions = model.predict(x_train, verbose = 0)
    train_cleaned_predictions = train_predictions.flatten().round()
    train_acc = np.mean(train_cleaned_predictions == y_train)
    train_mae = np.mean(np.abs(train_cleaned_predictions - y_train))

    print(f'True training accuracy: {train_acc*100:.4}')
    print(f'Training MAE: {train_mae:.4}')

    print("Generating true test accuracy")
    test_predictions = model.predict(x_test, verbose = 0)
    test_cleaned_predictions = test_predictions.flatten().round()
    test_acc = np.mean(test_cleaned_predictions == y_test)
    test_mae = np.mean(np.abs(test_cleaned_predictions - y_test))

    print(f'True test accuracy: {test_acc*100:.4}')
    print(f'Test MAE: {test_mae:.4}')

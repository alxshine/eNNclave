import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as layers

import utils
import mit_prepare_data

x_train, y_train = mit_prepare_data.load_train_set()
x_test, y_test = mit_prepare_data.load_test_set()

# generate datasets
train_ds = utils.generate_dataset(x_train, y_train, preprocess_function=None)
test_ds = utils.generate_dataset(
    x_test, y_test, shuffle=False, repeat=False, preprocess_function=None)


# build model
MODEL_FILE = 'models/mit_places.h5'
HIST_FILE = 'hist_mit_places.csv'
HIDDEN_NEURONS = 2048
DROPOUT_RATIO=0.4
NUM_EPOCHS = 2000
STEPS_PER_EPOCH = 3

full_model = load_model('models/vgg16_places365.h5')
extractor = Sequential(full_model.layers[:-9])
extractor.trainable = False

dense = Sequential([
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(HIDDEN_NEURONS, activation='relu'),
    layers.Dropout(DROPOUT_RATIO),
    layers.Dense(x_train.shape[1], activation='softmax')
])


model = Sequential([extractor,
                    layers.MaxPooling2D(2),
                    layers.Flatten(),
                    dense])

print('Hypeparameters:')
print('num_epochs: {}'.format(NUM_EPOCHS))
print('hidden_neurons: {}'.format(HIDDEN_NEURONS))
print('training set size: {}'.format(len(y_train)))
print('test set size: {}'.format(len(y_test)))
print()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_ds,
                    validation_steps=STEPS_PER_EPOCH)

loss0, accuracy0 = model.evaluate(test_ds)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
print("\nSaving model at: {}".format(MODEL_FILE))
model.save(MODEL_FILE)

print("Saving history at: {}".format(HIST_FILE))
hist_df = pd.DataFrame(history.history)
with open(HIST_FILE, 'w+') as f:
    hist_df.to_csv(f)

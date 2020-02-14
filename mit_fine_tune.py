import tensorflow.keras.layers as layers
import tensorflow.keras.applications as apps
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf

import pandas as pd
import sys

tf.compat.v1.set_random_seed(1337)

from os.path import join
import os
import json

import utils
import mit_prepare_data

# build model
if len(sys.argv) < 2:
    print('Usage: %s mit_model_file' % sys.argv[0])
    sys.exit(1)



MODEL_FILE = sys.argv[1]
model_basename = os.path.basename(MODEL_FILE).split('.')[0]

TARGET_MODEL_FILE = 'models/%s_tuned.h5' % model_basename
HIST_FILE = 'hist_%s_tuning.csv' % model_basename
NUM_EPOCHS = 200
STEPS_PER_EPOCH = 3
VALIDATION_STEPS = 3
UNFREEZE = False

model = load_model(MODEL_FILE)

if UNFREEZE:
    print('Unfreezing weights!')
    for l in model.layers:
        l.trainable = True
else:
    print('Keeping weights frozen')

optimizer = optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss=sparse_categorical_crossentropy,
              metrics=['accuracy'])

x_train, y_train = mit_prepare_data.load_train_set()
x_test, y_test = mit_prepare_data.load_test_set()

# generate datasets
train_ds = utils.generate_dataset(x_train, y_train, preprocess_function=None)
test_ds = utils.generate_dataset(
    x_test, y_test, shuffle=False, repeat=False, preprocess_function=None)


print('Hypeparameters:')
print('num_epochs: {}'.format(NUM_EPOCHS))
print('training set size: {}'.format(len(y_train)))
print('test set size: {}'.format(len(y_test)))
print()

model.summary()

loss0, accuracy0 = model.evaluate(test_ds)
print("Test set loss before warmstart fitting: %f, accuracy: %f" % (loss0, accuracy0))

history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_ds,
                    # validation_steps=VALIDATION_STEPS
                    )

loss1, accuracy1 = model.evaluate(test_ds)

print()
print("loss before: {:.2f}, after: {:.2f}".format(loss0,loss1))
print("accuracy: {:.2f}, after: {:.2f}".format(accuracy0, accuracy1))
print("\nSaving model at: {}".format(TARGET_MODEL_FILE))
model.save(TARGET_MODEL_FILE)

print("Saving history at: {}".format(HIST_FILE))
hist_df = pd.DataFrame(history.history)
with open(HIST_FILE, 'w+') as f:
    hist_df.to_csv(f)

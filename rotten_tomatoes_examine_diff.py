from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf
import numpy as np

from enclave_model import Enclave
from enclave_layer import EnclaveLayer
import utils
import interop.pymatutil as pymatutil

import os
from os.path import join

import json
import sys
import time

from rotten_tomatoes_prepare_data import load_rotten_tomatoes

if len(sys.argv) < 4:
    print("Usage: {} tf_model enclave_model sample index".format(sys.argv[0]))
    sys.exit(1)

_, _, x_test, y_test, _ = load_rotten_tomatoes('./datasets')

tf_model = load_model(sys.argv[1])
enclave_model = load_model(sys.argv[2], custom_objects={'EnclaveLayer': EnclaveLayer})

pymatutil.initialize()

i = int(sys.argv[3])

num_diffs = 0
sample = x_test[i:i+1]
    

enclave_predictions = enclave_model.predict(sample)
enclave_labels = np.argmax(enclave_predictions, axis=1)

sub_model = Sequential(tf_model.layers[:-6])
sub_model.build()
print("Sub model:")
sub_model.summary()
res = sub_model(sample).numpy()
print("Res.dtype: %s" % str(res.dtype))
print("Res: %s" % str(res.shape))
# for i in range(res.shape[1]):
for i in range(1):
    print(res[0,i,:])

pymatutil.teardown()

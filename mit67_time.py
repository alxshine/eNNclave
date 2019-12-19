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
import time

from mit67_prepare_data import load_test_set

def time_enclave_prediction(model, samples):
    # test if model has enclave part
    has_enclave = any(["enclave" in l.name for l in model.layers])

    # split model into TF and enclave part
    for enclave_start,l in enumerate(model.layers):
        if "enclave" in l.name:
            break

    tf_part = Sequential(model.layers[:enclave_start])
    enclave_part = model.layers[enclave_start]

    before = time.time()
    pymatutil.initialize()
    after_setup = time.time()

    # predict dataset
    tf_prediction = tf_part(samples)
    after_tf = time.time()

    final_prediction = enclave_part(tf_prediction)
    after_enclave = time.time()

    pymatutil.teardown()
    after_teardown = time.time()

    setup_time = after_setup - before
    gpu_time = after_tf - after_setup
    enclave_time = after_enclave - after_tf
    teardown_time = after_teardown - after_enclave
    total_time = after_teardown - before
    return {
        'setup_time': setup_time,
        'gpu_time': gpu_time,
        'enclave_time': enclave_time,
        'teardown_time': teardown_time
    }

if __name__ == '__main__':
    model_file = 'models/mit67_enclave.h5' # TODO: move to parameter
    model = load_model(model_file, custom_objects={'EnclaveLayer': EnclaveLayer})

    np.random.seed(1337)

    x_test, _ = load_test_set()
    sample_index = np.random.randint(x_test.shape[0])
    times = time_enclave_prediction(model, x_test[sample_index:sample_index+1])
    print(json.dumps(times, indent=2))

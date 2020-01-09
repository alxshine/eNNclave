from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf
import numpy as np

from enclave_model import Enclave
from enclave_layer import EnclaveLayer
import utils
import interop.pymatutil as pymatutil

import sys

import json
import time

from mit_prepare_data import load_test_set
from time_enclave import time_enclave_prediction

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage {} model".format(sys.argv[0]))
        sys.exit(1)
    
    model_file = sys.argv[1]
    model = load_model(model_file, custom_objects={'EnclaveLayer': EnclaveLayer})

    np.random.seed(1337)

    x_test, _ = load_test_set()
    #  sample_index = np.random.randint(x_test.shape[0])
    sample_index = 42
    times = time_enclave_prediction(model, x_test[sample_index:sample_index+1])
    print(json.dumps(times, indent=2))

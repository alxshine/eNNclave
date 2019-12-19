from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.backend import clear_session
import tensorflow as tf

import numpy as np
import time
import json
import multiprocessing

import interop.pymatutil as pymatutil
import enclave_layer

import build_enclave
import mit67_prepare_data

def time_from_file(model_path, samples, conn=None):
    model = load_model(model_path, custom_objects={'EnclaveLayer': enclave_layer.EnclaveLayer})
    time_dict = time_enclave_prediction(model, samples)
    
    #send output to parent process
    if conn != None:
        conn.send(time_dict)
    return time_dict

    
def time_enclave_prediction(model, samples):
    # test if model has enclave part
    has_enclave = any(["enclave" in l.name for l in model.layers])

    # split model into TF and enclave part
    for enclave_start,l in enumerate(model.layers):
        if "enclave" in l.name:
            break

    tf_part = Sequential(model.layers[:enclave_start])
    enclave_part = model.layers[enclave_start]

    # try and force tf to reinitialize

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

    enclave_setup_time = after_setup - before
    gpu_time = after_tf - after_setup
    enclave_time = after_enclave - after_tf
    teardown_time = after_teardown - after_enclave
    total_time = after_teardown - before

    time_dict = {
        'enclave_setup_time': enclave_setup_time,
        'gpu_time': gpu_time,
        'enclave_time': enclave_time,
        'teardown_time': teardown_time,
        'combined_enclave_time': enclave_time+enclave_setup_time+teardown_time
    }
    
    return time_dict


def time_cuts(model_path, cuts, samples):
    """ Generates a model for every cut and runs timing in subprocess """
    times = {}
    new_filename = build_enclave.get_new_filename(model_path)
    
    for cut in cuts:
        build_child = multiprocessing.Process(target=build_enclave.build_enclave,
                                              args=(model_path, cut))
        build_child.start()
        build_child.join()

        time_p_conn, time_c_conn = multiprocessing.Pipe()
        child = multiprocessing.Process(target=time_from_file, args=(new_filename, samples, time_c_conn))
        print("Timing cut at layer %d" % cut)
        child.start()
        times[cut] = time_p_conn.recv()
        child.join()
        
    return times

if __name__ == '__main__':
    original_file = 'models/mit67.h5'
    cuts = [1, 3, 5, 10, 14, 18, 21, 24]

    np.random.seed(1337)
    x_test, _ = mit67_prepare_data.load_test_set()
    sample_index = np.random.randint(x_test.shape[0])
    samples = x_test[sample_index:sample_index+1]
    
    times = time_cuts(original_file, cuts, samples)

    print("\n\n")
    print(json.dumps(times, indent=2))
    with open('timing_logs/mit67_times.json', 'w+') as f:
        json.dump(times, f, indent=2)

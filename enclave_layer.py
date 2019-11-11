import interop.pymatutil as pymatutil
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer


class EnclaveLayer(Layer):
    def wrap_matutil(self, xs):
        xs = xs.numpy()
        ret = np.zeros_like(xs)
        for i, x in enumerate(xs):
            label = pymatutil.dense(x.astype(np.float32).tobytes(),
                                    1, x.shape[0])
            ret[i, label] = 1
        return ret

    def call(self, inputs):
        return tf.py_function(func=self.wrap_matutil,
                              inp=[inputs], Tout=tf.float32)

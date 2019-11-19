import interop.pymatutil as pymatutil
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer


class EnclaveLayer(Layer):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super().__init__(**kwargs)

    def get_config(self):
        return {'num_classes': self.num_classes, 'name': super().name}
    
    def wrap_matutil(self, xs):
        xs = xs.numpy()
        ret = np.zeros((xs.shape[0],self.num_classes))
        for i, x in enumerate(xs):
            label = pymatutil.forward(x.astype(np.float32).tobytes(),
                                    1, x.shape[0])
            ret[i, label] = 1
        return ret

    def call(self, inputs):
        return tf.py_function(func=self.wrap_matutil,
                              inp=[inputs], Tout=tf.float32)

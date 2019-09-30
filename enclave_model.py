from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
import tensorflow as tf
from format_strings import *

class Enclave(Sequential):
    def __init__(self, layers=None, name='Enclave'):
        super().__init__(layers=layers, name=name)

    def generate_state(self, to_file_base='state'):
        file_template = to_file_base + "%s"
        header_file = open(file_template % ".hpp", "w+")
        cpp_file = open(file_template % ".cpp", 'w+')

        header_file.write("#ifndef STATE_H\n")
        header_file.write("#define STATE_H\n\n")

        for i, l in enumerate(self.layers):
            # TODO: test if layer is dense here, like in get_call_string
            parameters = l.get_weights()

            if len(parameters) > 0:
                w = parameters[0]

                lhs_string = "float w%d[]" % (i)
                header_file.write("extern " + lhs_string + ";\n")
                header_file.write("extern int w%d_r;\n" % i)
                header_file.write("extern int w%d_c;\n\n" % i)

                cpp_file.write(Enclave.dump_matrix(w, lhs_string))
                cpp_file.write("int w%d_r = %d;\n" % (i, w.shape[0]))
                cpp_file.write("int w%d_c = %d;\n\n" % (i, w.shape[1]))
            if len(parameters) > 1:
                b = parameters[1]
                lhs_string = "float b%d[]" % (i)
                header_file.write("extern " + lhs_string + ";\n")
                header_file.write("extern int b%d_c;\n\n" % i)

                cpp_file.write(Enclave.dump_matrix(b, lhs_string))
                cpp_file.write("int b%d_c = %d;\n\n" % (i, b.shape[0]))

        header_file.write("#endif\n")
        header_file.close()
        cpp_file.close()

    @staticmethod
    def dump_matrix(m, lhs_string):
        s = "%s = {\n" % lhs_string
        if len(m.shape) == 1:
            # 1D array
            for x in m:
                s += "%f, " % x
            s += "\n"
        else:
            # 2D array
            for row in m:
                for x in row:
                    s += "%f, " % x
                s += "\n"
        s += "};\n"
        return s

    def generate_dense(self, to_file='dense.cpp'):
        dense_file = open(to_file, 'w+')

        expected_c = self.layers[0].get_weights()[0].shape[0]
        preamble = preamble_template % (expected_c, expected_c)

        dense_file.write(preamble)
        increase_tmp_index = -1
        for i, l in enumerate(self.layers):
            if i == 0:
                inputs = 'm'
            else:
                inputs = tmp_buffer_template % (increase_tmp_index)

            call_string, increment_tmp_index = Enclave.get_call_string(
                inputs, i, l)
            dense_file.write(call_string)
            if increment_tmp_index:
                increase_tmp_index = i
        dense_file.write(postamble)
        dense_file.close()

    @staticmethod
    def get_call_string(inputs, tmp_index, layer):

        added_ops = True
        if type(layer) in [tf.keras.layers.Dense]:
            # the output of the dense layer will be a
            # row vector with ncols(w) elements
            tmp_name = tmp_buffer_template % tmp_index
            parameters = layer.get_weights()
            w = parameters[0]

            s = tmp_buffer_declaration_template % (
                tmp_index, w.shape[1])
            weight_name = weight_name_template % tmp_index
            mult = mult_template % (
                inputs, 1, w.shape[0], weight_name, w.shape[0], w.shape[1],
                tmp_name)
            s += error_handling_template % mult

            if len(parameters) > 1:
                # add bias
                b = parameters[1]
                bias_name = bias_name_template % tmp_index
                add = add_template % (
                    tmp_name, 1, w.shape[1], bias_name, 1, b.shape[0],
                    tmp_name)
                s += error_handling_template % add

            if layer.activation.__name__ == 'relu':  # TODO: do this cleaner
                relu = relu_template % (
                    tmp_name, 1, w.shape[1])
                s += relu
            elif layer.activation.__name__ == 'softmax':
                # here we compute the actual label
                softmax = softmax_template % (w.shape[1], tmp_name, tmp_name)
                s += softmax
            else:
                s += "ERROR: unknown activation function %s\n" % (
                    layer.activation.__name__)
        else:
            s = unknown_layer_template % (
                layer.name, type(layer))
            added_ops = False

        s += "\n"

        return s, added_ops

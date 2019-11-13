from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow as tf
from format_strings import *
import numpy as np


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
            if type(l) in [layers.Dense]:
                # TODO: test if layer is dense here, like in get_call_string
                parameters = l.get_weights()

                if len(parameters) > 0:
                    w = parameters[0]

                    lhs_string = "float *w%d" % (i)
                    header_file.write("extern " + lhs_string + ";\n")
                    header_file.write("extern int w%d_r;\n" % i)
                    header_file.write("extern int w%d_c;\n\n" % i)

                    with open('w%d.bin' % i, 'wb+') as f:
                        f.write(w.astype(np.float32).tobytes())

                    cpp_file.write(
                        "extern const char _binary_w%d_bin_start;\n" % i)
                    cpp_file.write(
                        "const float *w%d = (const float*) &_binary_w%d_bin_start;\n" % (i, i))
                    cpp_file.write("int w%d_r = %d;\n" % (i, w.shape[0]))
                    cpp_file.write("int w%d_c = %d;\n\n" % (i, w.shape[1]))
                if len(parameters) > 1:
                    b = parameters[1]
                    lhs_string = "float *b%d" % (i)
                    header_file.write("extern " + lhs_string + ";\n")
                    header_file.write("extern int b%d_c;\n\n" % i)

                    with open('b%d.bin' % i, 'wb+') as bf:
                        bf.write(b.astype(np.float32).tobytes())

                    cpp_file.write(
                        "extern const char _binary_b%d_bin_start;\n" % i)
                    cpp_file.write(
                        "const float *b%d = (const float*) &_binary_b%d_bin_start;\n" % (i, i))
                    cpp_file.write("int b%d_c = %d;\n\n" % (i, b.shape[0]))
            elif type(l) in [layers.Dropout]:
                # these layers are not used during inference
                continue
            else:
                raise NotImplementedError(
                    "Unknown layer type {}".format(type(l)))

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

    def generate_forward(self, to_file='forward.cpp'):
        forward_file = open(to_file, 'w+')

        # the first dim of input_shape is num_samples in batch, so skip that
        expected_c = self.layers[0].input_shape[1]
        preamble = preamble_template % (expected_c, expected_c)

        forward_file.write(preamble)
        increase_tmp_index = -1
        for i, l in enumerate(self.layers):
            if i == 0:
                inputs = 'm'
            else:
                inputs = tmp_buffer_template % (increase_tmp_index)

            call_string, increment_tmp_index = Enclave.get_call_string(
                inputs, i, l)
            forward_file.write(call_string)
            if increment_tmp_index:
                increase_tmp_index = i
        forward_file.write(postamble)
        forward_file.close()

    @staticmethod
    def get_call_string(inputs, tmp_index, layer):
        """Generates C function calls required for layer.

        Arguments:
        inputs -- the name of the input buffer
        tmp_index -- the index of the current tmp buffer
        layer -- the layer to generate for

        Returns:
        s -- the generated C code
        added_ops -- if actual code was generated
        """

        added_ops = True
        if type(layer) in [layers.Dense]:
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

            if layer.activation.__name__ == 'relu':
                relu = relu_template % (
                    tmp_name, 1, w.shape[1])
                s += relu
            elif layer.activation.__name__ == 'softmax':
                # here we compute the actual label
                softmax = softmax_template % (w.shape[1], tmp_name, tmp_name)
                s += softmax
            else:
                raise NotImplementedError("Unknown activation function {} in layer {}".format(
                    layer.activation.__name__, layer.name))
        elif type(layer) in [layers.Dropout]:
            # these layers are inactive during inference, so they can be skipped
            s = "//Layer {} skipped\n"
            return s, False
        else:
            raise NotImplementedError(
                "Unknown layer type {}".format(type(layer)))

        s += "\n"

        return s, added_ops

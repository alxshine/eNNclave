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
            if type(l) in [layers.Dense, layers.Conv2D]:
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
                    cpp_file.write("int w%d_h = %d;\n" % (i, w.shape[0]))
                    cpp_file.write("int w%d_w = %d;\n" % (i, w.shape[1]))
                    if len(w.shape) > 2:
                        cpp_file.write("int w%d_c = %d;\n" % (i, w.shape[2]))
                    if len(w.shape) > 3:
                        cpp_file.write("int w%d_f = %d;\n" % (i, w.shape[3]))
                    cpp_file.write("\n")

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

            elif type(l) in [layers.SeparableConv1D]:
                depth_kernels, point_kernels, biases = l.get_weights()

                #declare all arrays
                header_file.write("extern float *depth_kernels%d;\n" % i)
                header_file.write("extern float *point_kernels%d;\n" % i)
                header_file.write("extern float *b%d;\n" % i)

                #create binary files
                with open('depth%d.bin' %i, 'wb+') as df:
                    df.write(depth_kernels.astype(np.float32).tobytes())
                with open('point%d.bin' %i, 'wb+') as pf:
                    pf.write(point_kernels.astype(np.float32).tobytes())
                with open('b%d.bin' %i, 'wb+') as bf:
                    bf.write(biases.astype(np.float32).tobytes())

                #create nicer handles
                cpp_file.write(
                    "extern const char _binary_depth%d_bin_start;\n" %i)
                cpp_file.write(
                    "const float *depth_kernels%d = (const float*) &_binary_depth%d_bin_start;\n" % (i, i))
                cpp_file.write(
                    "extern const char _binary_point%d_bin_start;\n" %i)
                cpp_file.write(
                    "const float *point_kernels%d = (const float*) &_binary_point%d_bin_start;\n" % (i, i))
                cpp_file.write(
                    "extern const char _binary_b%d_bin_start;\n" %i)
                cpp_file.write(
                    "const float *b%d = (const float*) &_binary_b%d_bin_start;\n" % (i, i))
                
            elif type(l) in [layers.Dropout, layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D,
                             layers.MaxPooling1D,layers.MaxPooling2D]:
                # these layers are either not used during inference or have no parameters
                continue
            else:
                raise NotImplementedError(
                    "Unknown layer type {}".format(type(l)))

        header_file.write("#endif\n")
        header_file.close()
        cpp_file.close()

    def generate_forward(self, to_file='forward.cpp'):
        forward_file = open(to_file, 'w+')

        # the first dim of input_shape is num_samples in batch, so skip that
        expected_c = self.layers[0].input_shape[1]

        forward_file.write(preamble_template)
        # declare tmp buffers
        output_sizes = [np.prod(l.output_shape[1:]) for l in self.layers]
        output_sizes.sort(reverse = True)
        # TODO: check for even and odd layers to get optimal allocation size
        forward_file.write(tmp_buffer_declaration_template % (0, output_sizes[0]))
        forward_file.write(declaration_error_handling_template % (0, output_sizes[0]))
        forward_file.write(tmp_buffer_declaration_template % (1, output_sizes[0]))
        forward_file.write(declaration_error_handling_template % (1, output_sizes[0]))
            
        tmp_index = 0
        inputs = 'm'
        for i, l in enumerate(self.layers):
            call_string, generated_ops = Enclave.get_call_string(
                inputs, i, l, tmp_index)
            forward_file.write(call_string)

            #if the function generated a call, it declared a new tmp buffer
            if generated_ops:
                inputs = tmp_buffer_template % tmp_index
                tmp_index = 1-tmp_index

        #free tmp buffers
        forward_file.write(tmp_buffer_release_template % 0)
        forward_file.write(tmp_buffer_release_template % 1)
        forward_file.write(postamble)
        forward_file.close()

    @staticmethod
    def get_call_string(inputs, i, layer, tmp_index):
        """Generates C function calls required for layer.

        Arguments:
        inputs -- the name of the input buffer
        tmp_index -- the index of the current tmp buffer
        layer -- the layer to generate for

        Returns:
        s -- the generated C code
        added_ops -- True iff an operation was generated (as opposed to a comment)
        """

        added_ops = True
        if type(layer) in [layers.Dense]:
            # the output of the dense layer will be a
            # row vector with ncols(w) elements
            tmp_name = tmp_buffer_template % tmp_index
            parameters = layer.get_weights()
            w = parameters[0]

            weight_name = weight_name_template % i
            mult = mult_template % (
                inputs, 1, w.shape[0], weight_name, w.shape[0], w.shape[1],
                tmp_name)
            s = error_handling_template % mult

            if len(parameters) > 1:
                # add bias
                b = parameters[1]
                bias_name = bias_name_template % i
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
            elif layer.activation.__name__ == 'sigmoid':
                s += sigmoid_template % tmp_name
            else:
                raise NotImplementedError("Unknown activation function {} in layer {}".format(
                    layer.activation.__name__, layer.name))

        elif type(layer) in [layers.SeparableConv1D]:
            if layer.padding != 'same':
                raise NotImplementedError("Padding modes other than 'same' are not implemented")

            _, steps, c = layer.input_shape
            f = layer.output_shape[-1]
            new_size = np.prod(layer.output_shape[1:])
            new_buffer = tmp_buffer_template % tmp_index
            ks = layer.kernel_size[0]
            depth_kernels = depth_kernel_template % i
            point_kernels = point_kernel_template % i
            biases = bias_name_template % i

            s = sep_conv1_template % (inputs, steps, c, f, depth_kernels, point_kernels, ks, biases, new_buffer)

            if layer.activation.__name__ == 'relu':
                # relu
                s += relu_template % (new_buffer, 1, new_size)
            elif layer.activation.__name__ == 'linear':
                s += "  // no activation function for layer {}".format(layer.name)
            else:
                raise NotImplementedError("Unknown activation function {} in layer {}".format(
                    layer.activation.__name__, layer.name))
                        
        elif type(layer) in [layers.Conv2D]:
            if layer.padding != 'same':
                raise NotImplementedError("Padding modes other than 'same' are not implemented")
            
            _, h, w, c = layer.input_shape
            f = layer.output_shape[-1]
            new_size = np.prod(layer.output_shape[1:])
            new_buffer = tmp_buffer_template % tmp_index
            kh, kw = layer.kernel_size
            kernels = weight_name_template % i
            biases = bias_name_template % i
            
            s = conv2_template % (inputs, h, w, c, f, kernels, kh, kw, biases, new_buffer)

            if layer.activation.__name__ == 'relu':
                # relu
                s += relu_template % (new_buffer, 1, new_size)
            elif layer.activation.__name__ == 'linear':
                s += "  // no activation function for layer {}".format(layer.name)
            else:
                raise NotImplementedError("Unknown activation function {} in layer {}".format(
                    layer.activation.__name__, layer.name))

        elif type(layer) in [layers.GlobalAveragePooling1D]:
            _, steps, c = layer.input_shape
            s = global_average_pooling_1d_template % (inputs, steps, c, tmp_buffer_template % tmp_index)
            
        elif type(layer) in [layers.GlobalAveragePooling2D]:
            _, h, w, c = layer.input_shape
            s = global_average_pooling_2d_template % (inputs, h, w, c, tmp_buffer_template % tmp_index)

        elif type(layer) in [layers.MaxPooling1D]:
            _, steps, c = layer.input_shape
            s = max_pooling_1d_template % (inputs, steps, c, layer.pool_size[0], tmp_buffer_template % tmp_index)
        elif type(layer) in [layers.MaxPooling2D]:
            _, h, w, c = layer.input_shape
            pool_size = layer.pool_size[0]
            if layer.pool_size[0] != layer.pool_size[1]:
                raise NotImplementedError("Non-square pooling is not implemented")

            new_size = np.prod(layer.output_shape[1:])
            s = max_pooling_2d_template % (inputs, h, w, c, pool_size, tmp_buffer_template % tmp_index)

        elif type(layer) in [layers.Dropout]:
            # these layers are inactive during inference, so they can be skipped
            s = "//Layer {} skipped\n".format(layer.name)
            return s, False
        else:
            raise NotImplementedError(
                "Unknown layer type {}".format(type(layer)))

        s += "\n"

        return s, added_ops

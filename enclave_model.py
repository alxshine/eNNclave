from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow as tf
from format_strings import *
import numpy as np
import os

import utils


class Enclave(Sequential):
    def __init__(self, layers=None, name='Enclave'):
        super().__init__(layers=layers, name=name)

    def generate_config(self, target_dir='lib/enclave/enclave/'):
        all_layers = utils.get_all_layers(self)
        output_sizes = [np.prod(l.output_shape[1:]) for l in all_layers]
        output_sizes.sort(reverse = True)
        # get max tmp_buffer size
        max_size = output_sizes[0]
        total_tmp_size = 2*max_size*4
        # align to 4kB
        num_heap_blocks = int(np.ceil(total_tmp_size / 0x1000))
        num_heap_blocks += 1000 # for tolerance
        heap_size = num_heap_blocks * 0x1000

        print("Max required heap size: %s MB" % (heap_size/1024/1024))
        # override for now
        mb_size = 126
        print("Configuring heap size for %d MB for now" % mb_size)
        heap_size = mb_size*1024*1024
        config_path = os.path.join(target_dir, 'config.xml')
        config = config_template % (hex(heap_size), hex(heap_size))

        with open(config_path, 'w+') as config_file:
            config_file.write(config)

    def generate_state(self):
        bin_file = 'parameters.bin'
        bf = open(bin_file, 'w+b')

        for i, l in enumerate(self.layers):
            if type(l) in [layers.Dense, layers.Conv2D]:
                parameters = l.get_weights()

                if len(parameters) > 0:
                    w = parameters[0]
                    bf.write(w.astype(np.float32).tobytes())

                if len(parameters) > 1:
                    b = parameters[1]
                    bf.write(b.astype(np.float32).tobytes())

            elif type(l) in [layers.SeparableConv1D]:
                depth_kernels, point_kernels, biases = l.get_weights()

                bf.write(depth_kernels.astype(np.float32).tobytes())
                bf.write(point_kernels.astype(np.float32).tobytes())
                bf.write(biases.astype(np.float32).tobytes())

            elif type(l) in [layers.Dropout, layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D,
                             layers.MaxPooling1D,layers.MaxPooling2D, layers.Flatten]:
                # these layers are either not used during inference or have no parameters
                continue
            else:
                raise NotImplementedError(
                    "Unknown layer type {}".format(type(l)))

        bf.close()

    def generate_forward(self, to_file='forward.c', target_dir='lib/enclave/enclave'):
        target_file = os.path.join(target_dir, to_file)
        forward_file = open(target_file, 'w+')
        all_layers = utils.get_all_layers(self)

        # the first dim of input_shape is num_samples in batch, so skip that
        expected_c = self.layers[0].input_shape[1]

        parent_dir = target_dir.split('/')[-1]
        forward_file.write(preamble_template % parent_dir)
        # declare tmp buffers
        output_sizes = [np.prod(l.output_shape[1:]) for l in all_layers]
        output_sizes.sort(reverse = True)

        # get required size for weight buffer
        param_numbers = [np.sum([np.prod(w.shape) for w in l.get_weights()]) for l in all_layers]
        max_size = max(param_numbers)
        forward_file.write(buffer_declaration_template % (output_sizes[0], output_sizes[0], output_sizes[0], output_sizes[0], max_size, max_size))

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
        forward_file.write(release_template)
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
        s = ''
        if type(layer) in [layers.Dense]:
            # the output of the dense layer will be a
            # row vector with ncols(w) elements
            tmp_name = tmp_buffer_template % tmp_index
            parameters = layer.get_weights()
            num_params = [np.prod(p.shape) for p in parameters]
            s += load_template % np.sum(num_params)
            w = parameters[0]

            weight_name = parameter_offset_template % 0
            mult = mult_template % (
                inputs, 1, w.shape[0], weight_name, w.shape[0], w.shape[1],
                tmp_name)
            s += error_handling_template % mult

            if len(parameters) > 1:
                # add bias
                b = parameters[1]
                bias_name = parameter_offset_template % num_params[0]
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
            elif layer.activation.__name__ == 'linear':
                s += '\t//linear activation requires no action\n'
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

            # from matutil:
            # num_depth = ks*c
            # num_point = c*f
            # num_bias = f
            s += load_template % (ks*c+c*f+f)

            depth_kernels = parameter_offset_template % 0
            point_kernels = parameter_offset_template % (ks*c)
            biases = parameter_offset_template % (ks*c+c*f)
            s += sep_conv1_template % (inputs, steps, c, f, depth_kernels, point_kernels, ks, biases, new_buffer)

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

            s += load_template % (kw*kh*c*f + f) 
            kernels = parameter_offset_template % 0
            biases = parameter_offset_template % (kw*kh*c*f)

            s += conv2_template % (inputs, h, w, c, f, kernels, kh, kw, biases, new_buffer)

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

        elif type(layer) in [layers.Dropout, layers.Flatten]:
            # these layers are inactive during inference, so they can be skipped
            s = "//Layer {} skipped\n".format(layer.name)
            return s, False
        else:
            raise NotImplementedError(
                "Unknown layer type {}".format(type(layer)))

        s += "\n"

        return s, added_ops

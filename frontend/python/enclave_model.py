from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from frontend.python import templates, utils
import numpy as np
import os


class Enclave(Sequential):
    def __init__(self, layers=None, name='Enclave'):
        super().__init__(layers=layers, name=name)

    def generate_config(self, target_dir='backend/generated'):
        all_layers = utils.get_all_layers(self)
        output_sizes = [np.prod(l.output_shape[1:]) for l in all_layers]
        output_sizes.sort(reverse=True)
        # get max tmp_buffer size
        max_size = output_sizes[0]
        total_tmp_size = 2 * max_size * 4
        # align to 4kB
        num_heap_blocks = int(np.ceil(total_tmp_size / 0x1000))
        num_heap_blocks += 1000  # for tolerance
        heap_size = num_heap_blocks * 0x1000

        # print("Max required heap size: %s MB" % (heap_size/1024/1024))
        # override for now
        mb_size = 126
        # print("Configuring heap size for %d MB for now" % mb_size)
        heap_size = mb_size * 1024 * 1024
        config_path = os.path.join(target_dir, 'config.xml')
        config = templates.config.render(
            heapInitSize=hex(heap_size), heapMaxSize=hex(heap_size))

        with open(config_path, 'w+') as config_file:
            config_file.write(config)

    def generate_state(self, target_dir='backend/generated'):
        bin_file = os.path.join(target_dir, 'parameters.bin')
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

            elif type(l) in [layers.DepthwiseConv2D]:
                depth_kernels = l.get_weights()[0]

                bf.write(depth_kernels.astype(np.float32).tobytes())

            elif type(l) in [layers.Dropout, layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D,
                             layers.MaxPooling1D, layers.MaxPooling2D, layers.Flatten, layers.ZeroPadding2D,
                             layers.ReLU]:
                # these layers are either not used during inference or have no parameters
                continue
            else:
                raise NotImplementedError(
                    "Unknown layer type {}".format(type(l)))

        bf.close()

    def generate_forward(self, backend, target_dir='backend/generated'):
        target_file = os.path.join(target_dir, f'{backend}_forward.c')
        forward_file = open(target_file, 'w+')
        all_layers = utils.get_all_layers(self)

        forward_file.write(templates.preamble.render(backend=backend))
        # declare tmp buffers
        output_sizes = [np.prod(l.output_shape[1:]) for l in all_layers]
        output_sizes.sort(reverse=True)

        # get required size for weight buffer
        param_numbers = [np.sum([np.prod(w.shape)
                                 for w in l.get_weights()]) for l in all_layers]
        max_size = max(param_numbers)
        forward_file.write(templates.buffer_declaration.render(
            tmp1_size=output_sizes[0], tmp2_size=output_sizes[0], param_size=max_size))

        tmp_index = 0
        inputs = 'm'
        for i, l in enumerate(self.layers):
            tmp_name = templates.tmp_buffer.render(i=tmp_index)
            call_string, generated_ops = Enclave.get_call_string(
                inputs, l, tmp_name)
            forward_file.write(call_string)

            # if the function generated a call, it switched tmp buffers
            # TODO: always switch buffers
            if generated_ops:
                inputs = tmp_name
                tmp_index = 1 - tmp_index
                tmp_name = templates.tmp_buffer.render(i=tmp_index)

        # set result buffer
        forward_file.write(templates.return_results.render(input=inputs))
        # free tmp buffers
        forward_file.write(templates.release_buffers)
        forward_file.write(templates.postamble)
        forward_file.close()

    @staticmethod
    def get_call_string(inputs, layer, tmp_name):
        """Generates C function calls required for layer.

        Arguments:
        inputs -- the name of the input buffer
        tmp_name -- the name of the current tmp buffer
        layer -- the layer to generate for

        Returns:
        s -- the generated C code
        added_ops -- True iff an operation was generated (as opposed to a comment)
        """

        added_ops = True
        s = ''
        if type(layer) in [layers.Dense]:
            s += Enclave.generate_dense(layer, inputs, tmp_name)

        elif type(layer) in [layers.SeparableConv1D]:
            s += Enclave.generate_separable_conv1d(layer, inputs, tmp_name)

        elif type(layer) in [layers.Conv2D]:
            s += Enclave.generate_conv_2d(inputs, layer, tmp_name)

        elif type(layer) in [layers.DepthwiseConv2D]:
            s += Enclave.generate_depthwise_conv_2d(inputs, layer, tmp_name)

        elif type(layer) in [layers.GlobalAveragePooling1D]:
            _, steps, c = layer.input_shape
            s = templates.global_average_pooling_1d.render(
                input=inputs,
                steps=steps,
                channels=c,
                ret=tmp_name)

        elif type(layer) in [layers.GlobalAveragePooling2D]:
            _, h, w, c = layer.input_shape
            s = templates.global_average_pooling_2d.render(
                input=inputs,
                h=h,
                w=w,
                channels=c,
                ret=tmp_name)

        elif type(layer) in [layers.MaxPooling1D]:
            _, steps, c = layer.input_shape
            s = templates.max_pooling_1d.render(
                input=inputs,
                steps=steps,
                channels=c,
                pool_size=layer.pool_size[0],
                ret=tmp_name)

        elif type(layer) in [layers.MaxPooling2D]:
            _, h, w, c = layer.input_shape
            pool_size = layer.pool_size[0]
            if layer.pool_size[0] != layer.pool_size[1]:
                raise NotImplementedError(
                    "Non-square pooling is not implemented")

            s = templates.max_pooling_2d.render(
                input=inputs,
                h=h,
                w=w,
                channels=c,
                pool_size=pool_size,
                ret=tmp_name)

        elif type(layer) in [layers.ZeroPadding2D]:
            _, h, w, c = layer.input_shape
            padding = layer.padding
            if len(padding) != 2:
                raise NotImplementedError(
                    "Asymmetrical padding is not implemented")

            s = templates.zero_pad2.render(
                input=inputs,
                h=h,
                w=w,
                channels=c,
                top_pad=padding[0][0],
                bottom_pad=padding[0][1],
                left_pad=padding[1][0],
                right_pad=padding[1][1],
                ret=tmp_name)

        elif type(layer) in [layers.ReLU]:
            size = np.prod(layer.output_shape[1:])
            s += templates.relu.render(input=inputs, h=1, w=size)

            added_ops = False

        elif type(layer) in [layers.Dropout, layers.Flatten]:
            # these layers are inactive during inference, so they can be skipped
            s = "//Layer {} skipped\n".format(layer.name)
            return s, False
        else:
            raise NotImplementedError(
                "Unknown layer type {}".format(type(layer)))

        s += "\n"

        return s, added_ops

    @staticmethod
    def generate_dense(layer, inputs, tmp_name):
        # the output of the dense layer will be a
        # row vector with ncols(w) elements
        ret = ''

        parameters = layer.get_weights()
        num_params = [np.prod(p.shape) for p in parameters]
        ret += templates.load.render(num_params=np.sum(num_params))
        w = parameters[0]

        weight_name = templates.parameter_offset.render(offset=0)
        mult = templates.multiply.render(
            m1=inputs, h1=1, w1=w.shape[0], m2=weight_name, h2=w.shape[0], w2=w.shape[1],
            ret=tmp_name)
        ret += templates.handle_error.render(expression=mult)

        if len(parameters) > 1:
            # add bias
            b = parameters[1]
            bias_name = templates.parameter_offset.render(
                offset=num_params[0])
            add = templates.add.render(
                m1=tmp_name, h1=1, w1=w.shape[1], m2=bias_name, h2=1, w2=b.shape[0],
                ret=tmp_name)
            ret += templates.handle_error.render(expression=add)

        ret += Enclave.generate_activation(layer, tmp_name, w.shape[1])

        return ret

    @staticmethod
    def generate_separable_conv1d(layer, inputs, tmp_name):
        if layer.padding != 'same':
            raise NotImplementedError(
                "Padding modes other than 'same' are not implemented")

        ret = ''
        _, steps, c = layer.input_shape
        f = layer.output_shape[-1]

        new_size = np.prod(layer.output_shape[1:])
        ks = layer.kernel_size[0]

        # from nn:
        # num_depth = ks*c
        # num_point = c*f
        # num_bias = f
        ret += templates.load.render(num_params=ks * c + c * f + f)

        depth_kernels = templates.parameter_offset.render(offset=0)
        point_kernels = templates.parameter_offset.render(offset=ks * c)
        biases = templates.parameter_offset.render(offset=ks * c + c * f)
        ret += templates.sep_conv1.render(
            input=inputs,
            steps=steps,
            channels=c,
            filters=f,
            depth_kernels=depth_kernels,
            point_kernels=point_kernels,
            kernel_size=ks,
            biases=biases,
            ret=tmp_name)

        ret += Enclave.generate_activation(layer, tmp_name, new_size)
        return ret

    @staticmethod
    def generate_depthwise_conv_2d(inputs, layer, tmp_name):
        if layer.padding == 'valid':
            padding = 'PADDING_VALID'
        elif layer.padding == 'same':
            padding = 'PADDING_SAME'
        else:
            raise NotImplementedError(
                f"Padding {layer.padding} not implemented, requested for layer {layer}")

        ret = ''
        _, h, w, c = layer.input_shape
        f = layer.output_shape[-1]
        new_size = np.prod(layer.output_shape[1:])
        kh, kw = layer.kernel_size

        ret += templates.load.render(num_params=kw * kh * c * f + f)
        kernels = templates.parameter_offset.render(offset=0)

        ret += templates.depthwise_conv2.render(
            input=inputs,
            h=h,
            w=w,
            channels=c,
            padding=padding,
            kernels=kernels,
            kernel_height=kh,
            kernel_width=kw,
            ret=tmp_name)

        ret += Enclave.generate_activation(layer, tmp_name, new_size)

        return ret

    @staticmethod
    def generate_conv_2d(inputs, layer, tmp_name):
        if layer.padding != 'same':
            raise NotImplementedError(
                "Padding modes other than 'same' are not implemented")

        ret = ''
        _, h, w, c = layer.input_shape
        f = layer.output_shape[-1]
        new_size = np.prod(layer.output_shape[1:])
        kh, kw = layer.kernel_size

        ret += templates.load.render(num_params=kw * kh * c * f + f)
        kernels = templates.parameter_offset.render(offset=0)
        biases = templates.parameter_offset.render(offset=kw * kh * c * f)

        ret += templates.conv2.render(
            input=inputs,
            h=h,
            w=w,
            channels=c,
            filters=f,
            kernels=kernels,
            kernel_height=kh,
            kernel_width=kw,
            biases=biases,
            ret=tmp_name)

        ret += Enclave.generate_activation(layer, tmp_name, new_size)

        return ret

    @staticmethod
    def generate_activation(layer, target_buffer, input_size):
        if layer.activation.__name__ == 'relu':
            relu = templates.relu.render(
                m=target_buffer, h=1, w=input_size)
            return relu
        elif layer.activation.__name__ == 'softmax':
            # here we compute the actual label
            softmax = templates.softmax.render(
                num_labels=input_size, input=target_buffer)
            return softmax
        elif layer.activation.__name__ == 'sigmoid':
            return templates.sigmoid.render(input=target_buffer)
        elif layer.activation.__name__ == 'linear':
            return '\t//linear activation requires no action\n'
        else:
            raise NotImplementedError("Unknown activation function {} in layer {}".format(
                layer.activation.__name__, layer.name))
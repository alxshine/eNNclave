from tensorflow.keras.models import Sequential
import tensorflow as tf


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
                lhs_string = "float w%d[]" % i
                header_file.write(lhs_string + ";\n")
                header_file.write("int w%d_r;\n" % i)
                header_file.write("int w%d_c;\n\n" % i)

                cpp_file.write(Enclave.dump_matrix(w, lhs_string))
                cpp_file.write("int w%d_r = %d;\n" % (i, w.shape[0]))
                cpp_file.write("int w%d_c = %d;\n\n" % (i, w.shape[1]))
            if len(parameters) > 1:
                b = parameters[1]
                lhs_string = "float b%d[]" % i
                header_file.write(lhs_string + ";\n")
                header_file.write("int b%d_c;\n\n" % i)

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

        dense = """
int matutil_dense(float *m, int r, int c, int *label) {
  if (r != 1 || c != w1_r) {
    fprintf(stderr, "Input should be 1x%d\\n", w1_r);
    return -1;
  }
  int sts;
"""

        dense_file.write(dense)
        last_tmp_index = -1
        for i, l in enumerate(self.layers):
            if i == 0:
                inputs = 'm'
            else:
                inputs = 'tmp%d' % (last_tmp_index)

            call_string = Enclave.get_call_string(
                inputs, i, l)
            dense_file.write(call_string)
            last_tmp_index = i

        dense_file.close()

    @staticmethod
    def get_call_string(inputs, tmp_index, layer):
        error_handling = """if ((sts = %s))
  return sts;
"""

        if type(layer) in [tf.keras.layers.Dense]:
            # the output of the dense layer will be a
            # row vector with ncols(w) elements
            tmp_name = "tmp%d" % tmp_index
            parameters = layer.get_weights()
            w = parameters[0]

            s = "float %s[%d];\n" % (
                tmp_name, w.shape[1])
            weight_name = "w%d" % tmp_index
            mult = "matutil_multiply(%s, %d, %d, %s, %d, %d, %s)" % (
                inputs, 1, w.shape[0], weight_name, w.shape[0], w.shape[1],
                tmp_name)
            s += error_handling % mult
            
        else:
            s = "//No call method generated for layer %s of type %s\n" % (
                layer.name, type(layer))

        return s

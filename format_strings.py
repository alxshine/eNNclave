preamble_template = """
#include <stdlib.h>
#include <math.h>

#include "state.hpp"
#include "native.hpp"
#include "enclave.hpp"
#include "forward.hpp"

int forward(float *m, int, int *label) {
  int sts;

"""
postamble = """
  return 0;
}
"""


tmp_buffer_template = 'tmp%d'
tmp_buffer_declaration_template = "  float *tmp%d = (float*) malloc(%d*sizeof(float));\n"
declaration_error_handling_template = """  if(tmp%d == NULL){
  print_error("\\n\\nENCLAVE ERROR:Could not allocate buffer of size %d\\n\\n\\n");
  return 1;
  }
"""
tmp_buffer_release_template = "  free(tmp%d);\n"

weight_name_template = "weights%d"
depth_kernel_template = "depth_kernels%d"
point_kernel_template = "point_kernels%d"
bias_name_template = "biases%d"

error_handling_template = """  if ((sts = %s))
    return sts;
"""
add_template = "matutil_add(%s, %d, %d, %s, %d, %d, %s)"
mult_template = "matutil_multiply(%s, %d, %d, %s, %d, %d, %s)"
sep_conv1_template = "  matutil_sep_conv1(%s, %d, %d, %d, %s, %s, %d, %s, %s);\n"
conv2_template = "  matutil_conv2(%s, %d, %d, %d, %d, %s, %d, %d, %s, %s);\n"
relu_template = "  matutil_relu(%s, %d, %d);\n"
softmax_template = """
  // get maximum for label
  int max_index = 0;
  for (int i = 1; i < %d; ++i)
    max_index = %s[i] > %s[max_index] ? i : max_index;

  *label = max_index;
"""
sigmoid_template = """
  // fake sigmoid
  *label = %s[0] > 0.5;"""
global_average_pooling_1d_template = "  matutil_global_average_pooling_1d(%s, %d, %d, %s);\n"
global_average_pooling_2d_template = "  matutil_global_average_pooling_2d(%s, %d, %d, %d, %s);\n"
max_pooling_1d_template = "  matutil_max_pooling_1d(%s, %d, %d, %d, %s);\n"
max_pooling_2d_template = "  matutil_max_pooling_2d(%s, %d, %d, %d, %d, %s);\n"
unknown_layer_template = "  //No call method generated for layer %s of type %s\n"

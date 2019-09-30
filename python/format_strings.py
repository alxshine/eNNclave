preamble_template = """
#include "state.hpp"
#include "matutil.hpp"

int matutil_dense(float *m, int r, int c, int *label) {
  if (r != 1 || c != %d) {
    fprintf(stderr, "ERROR: Input should be 1x%d, got %%dx%%d\\n", r, c);
    return -1;
  }
  int sts;

"""
postamble = """
  return 0;
}
"""


tmp_buffer_template = 'tmp%d'
tmp_buffer_declaration_template = "  float tmp%d[%d];\n"

weight_name_template = "w%d"
bias_name_template = "b%d"

error_handling_template = """  if ((sts = %s))
    return sts;
"""
add_template = "matutil_add(%s, %d, %d, %s, %d, %d, %s)"
mult_template = "matutil_multiply(%s, %d, %d, %s, %d, %d, %s)"
relu_template = "  matutil_relu(%s, %d, %d);\n"
softmax_template = """
  // get maximum for label
  int max_index = 0;
  for (int i = 1; i < %d; ++i)
    max_index = %s[i] > %s[max_index] ? i : max_index;

  *label = max_index;
"""
unknown_layer_template = "  //No call method generated for layer %s of type %s\n"

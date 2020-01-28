preamble_template = """
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "forward.h"

#include "matutil.h"
#include "parameters.h"
#include "enclave_nn.h"
#include "native_nn.h"

int %s_f(float *m, int s, int *label) {
    int sts;

    open_parameters();
"""
postamble = """
    close_parameters();
    return 0;
}
"""

tmp_buffer_template = "tmp%d"
buffer_declaration_template = """
    float *tmp0 = (float*) malloc(%d*sizeof(float));
    if(tmp0 == NULL){
        print_error("\\n\\nENCLAVE ERROR:Could not allocate tmp0 of size %d\\n\\n\\n");
        return 1;
    }

    float *tmp1 = (float*) malloc(%d*sizeof(float));
    if(tmp1 == NULL){
        print_error("\\n\\nENCLAVE ERROR:Could not allocate tmp1 of size %d\\n\\n\\n");
        return 1;
    }

    float *params = (float*) malloc(%d*sizeof(float));
    if(params == NULL){
        print_error("\\n\\nENCLAVE ERROR:Could not allocate params of size %d\\n\\n\\n");
        return 1;
    }
"""
release_template = """
    free(tmp0);
    free(tmp1);
    free(params);
"""

weight_name_template = "weights%d"
depth_kernel_template = "depth_kernels%d"
point_kernel_template = "point_kernels%d"
bias_name_template = "biases%d"

error_handling_template = """  if ((sts = %s))
    return sts;
"""
load_template = "   load_parameters(params, %d);\n"
parameter_offset_template = 'params+%d'
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

config_template = """
<EnclaveConfiguration>
  <ProdID>0</ProdID>
  <ISVSVN>0</ISVSVN>
  <StackMaxSize>0x40000</StackMaxSize>
  <HeapInitSize>%s</HeapInitSize>
  <HeapMaxSize>%s</HeapMaxSize>
  <TCSNum>10</TCSNum>
  <TCSPolicy>1</TCSPolicy>
  <!-- Recommend changing 'DisableDebug' to 1 to make the enclave undebuggable for enclave release -->
  <DisableDebug>0</DisableDebug>
  <MiscSelect>0</MiscSelect>
  <MiscMask>0xFFFFFFFF</MiscMask>
</EnclaveConfiguration>
"""

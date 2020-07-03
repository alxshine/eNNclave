from jinja2 import Template

preamble = Template("""
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "forward.h"

#include "matutil.h"
#include "parameters.h"
#include "enclave_nn.h"
#include "native_nn.h"
#include "output.h"

int {{ mode }}_f(float *m, int s, float *ret, int rs) {
    int sts;

    open_parameters();
""")
postamble = """
    close_parameters();
    return 0;
}
"""

tmp_buffer = Template("tmp{{i}}")
buffer_declaration = Template("""
    float *tmp0 = (float*) malloc({{ tmp1_size }}*sizeof(float));
    if(tmp0 == NULL){
        print_err("\\n\\nENCLAVE ERROR:Could not allocate tmp0 of size {{ tmp1_size }}\\n\\n\\n");
        return 1;
    }

    float *tmp1 = (float*) malloc({{ tmp2_size }}*sizeof(float));
    if(tmp1 == NULL){
        print_err("\\n\\nENCLAVE ERROR:Could not allocate tmp1 of size {{ tmp2_size }}\\n\\n\\n");
        return 1;
    }

    float *params = (float*) malloc({{ param_size }}*sizeof(float));
    if(params == NULL){
        print_err("\\n\\nENCLAVE ERROR:Could not allocate params of size {{ param_size }}\\n\\n\\n");
        return 1;
    }
""")

release_buffers = """
    free(tmp0);
    free(tmp1);
    free(params);
"""
return_results = Template("""
  for(int i=0; i<rs; ++i)
    ret[i] = {{input}}[i];
  """)

handle_error = Template("""  if ((sts = {{expression}}))
    return sts;
""") # TODO: is this really necessary or useful?
load = Template("   load_parameters(params, {{num_params}});\n")
parameter_offset = Template("params+{{offset}}")
add = Template("matutil_add({{m1}}, {{h1}}, {{w1}}, {{m2}}, {{h2}}, {{w2}}, {{ret}})")
multiply = Template("matutil_multiply({{m1}}, {{h1}}, {{w1}}, {{m2}}, {{h2}}, {{w2}}, {{ret}})")
sep_conv1 = Template("  matutil_sep_conv1({{input}}, {{steps}}, {{channels}}, {{filters}}, {{depth_kernels}}, {{point_kernels}}, {{kernel_size}}, {{biases}}, {{ret}});\n")
conv2 = Template("  matutil_conv2({{input}}, {{h}}, {{w}}, {{channels}}, {{filters}}, {{kernels}}, {{kernel_height}}, {{kernel_width}}, {{biases}}, {{ret}});\n")
depthwise_conv2 = Template("  matutil_depthwise_conv2({{input}}, {{h}}, {{w}}, {{channels}}, {{padding}}, {{kernels}}, {{kernel_height}}, {{kernel_width}}, {{ret}});\n")
relu = Template("  matutil_relu({{m}}, {{h}}, {{w}});\n")
zero_pad2 = Template("  matutil_zero_pad2({{input}}, {{h}}, {{w}}, {{c}}, {{top_pad}}, {{bottom_pad}}, {{left_pad}}, {{right_pad}}, {{ret}});\n")
softmax = Template("""
  // get maximum for label
  int max_index = 0;
  for (int i = 1; i < {{num_labels}}; ++i)
    max_index = {{input}}[i] > {{input}}[max_index] ? i : max_index;

  for(int i=0; i< {{num_labels}}; ++i)
    {{input}}[i] = i == max_index ? 1 : 0;
""") # TODO: make this single pass
sigmoid = Template("""
  // fake sigmoid
  ret[0] = {{input}}[0] > 0.5;""")
global_average_pooling_1d = Template("  matutil_global_average_pooling_1d({{input}}, {{steps}}, {{channels}}, {{ret}});\n")
global_average_pooling_2d = Template("  matutil_global_average_pooling_2d({{input}}, {{h}}, {{w}}, {{channels}}, {{ret}});\n")
max_pooling_1d = Template("  matutil_max_pooling_1d({{input}}, {{steps}}, {{channels}}, {{pool_size}}, {{ret}});\n")
max_pooling_2d = Template("  matutil_max_pooling_2d({{input}}, {{h}}, {{w}}, {{channels}}, {{pool_size}}, {{ret}});\n")
dump = Template("  matutil_dump_matrix({{inputs}}, {{h}}, {{w}});")
dump3 = Template("  matutil_dump_matrix3({{inputs}}, {{h}}, {{w}}, {{channels}});")

config = Template("""
<EnclaveConfiguration>
  <ProdID>0</ProdID>
  <ISVSVN>0</ISVSVN>
  <StackMaxSize>0x40000</StackMaxSize>
  <HeapInitSize>{{heapInitSize}}</HeapInitSize>
  <HeapMaxSize>{{heapMaxSize}}</HeapMaxSize>
  <TCSNum>10</TCSNum>
  <TCSPolicy>1</TCSPolicy>
  <!-- Recommend changing 'DisableDebug' to 1 to make the enclave undebuggable for enclave release -->
  <DisableDebug>0</DisableDebug>
  <MiscSelect>0</MiscSelect>
  <MiscMask>0xFFFFFFFF</MiscMask>
</EnclaveConfiguration>
""")
from jinja2 import Template

preamble = Template("""
#include <cstdlib>
#include "backends.h"

#include "nn.h"
#include "IParameterLoader.h"
#include "output.h"

using namespace eNNclave;

#if defined(__cplusplus)
extern "C" {
#endif
int {{ backend }}_forward(float *m, int s, float *ret, int rs) {
    auto parameterLoader = getParameterLoader("{{parameter_file}}");
""")
postamble = """
    return 0;
}
#if defined(__cplusplus)
}
#endif
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
""") # TODO: make this nice C++

release_buffers = """
    free(tmp0);
    free(tmp1);
    free(params);
"""
return_results = Template("""
  for(int i=0; i<rs; ++i)
    ret[i] = {{input}}[i];
  """)

load = Template("parameterLoader->LoadParameters(params, {{num_params}});\n")
parameter_offset = Template("params+{{offset}}")
dense = Template("dense({{input}}, {{h}}, {{w}}, {{weights}}, {{neurons}}, {{biases}}, {{ret}});\n")
global_average_pooling_1d = Template("global_average_pooling_1d({{input}}, {{steps}}, {{channels}}, {{ret}});\n")
global_average_pooling_2d = Template("global_average_pooling_2d({{input}}, {{h}}, {{w}}, {{channels}}, {{ret}});\n")
max_pooling_1d = Template("max_pooling_1d({{input}}, {{steps}}, {{channels}}, {{pool_size}}, {{ret}});\n")
max_pooling_2d = Template("max_pooling_2d({{input}}, {{h}}, {{w}}, {{channels}}, {{pool_size}}, {{ret}});\n")
zero_pad2 = Template(
    """zero_pad2({{input}}, {{h}}, {{w}}, {{channels}}, {{top_pad}}, {{bottom_pad}},
    {{left_pad}}, {{right_pad}}, {{ret}});\n""")
relu = Template("relu({{m}}, {{size}});\n")
sep_conv1 = Template("""sep_conv1({{input}}, {{steps}}, {{channels}}, {{filters}}, {{depth_kernels}},
    {{point_kernels}}, {{kernel_size}}, {{biases}}, {{ret}});\n""")
depthwise_conv2 = Template("""depthwise_conv2({{input}}, {{h}}, {{w}}, {{channels}}, {{padding}}, {{kernels}},
    {{kernel_height}}, {{kernel_width}}, {{ret}});\n""")
conv2 = Template("""conv2({{input}}, {{h}}, {{w}}, {{channels}}, {{filters}}, {{kernels}}, {{kernel_height}},
    {{kernel_width}}, {{biases}}, {{ret}});\n""")

config = Template("""
<EnclaveConfiguration>
  <ProdID>0</ProdID>
  <ISVSVN>0</ISVSVN>
  <StackMaxSize>0x40000</StackMaxSize>
  <HeapInitSize>{{heapInitSize}}</HeapInitSize>
  <HeapMaxSize>{{heapMaxSize}}</HeapMaxSize>
  <TCSNum>10</TCSNum>
  <TCSPolicy>1</TCSPolicy>
  <!-- Recommend changing 'DisableDebug' to 1 to make the sgx undebuggable for sgx release -->
  <DisableDebug>0</DisableDebug>
  <MiscSelect>0</MiscSelect>
  <MiscMask>0xFFFFFFFF</MiscMask>
</EnclaveConfiguration>
""")

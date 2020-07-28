from jinja2 import Template

preamble = Template("""
#include <cstdlib>
#include "backend_{{ backend }}.h"

#include "nxx.h"
#include "{{PascalCaseBackend}}ParameterLoader.h"
#include "output.h"

#ifdef _cplusplus
extern "C" {
#endif
int {{ backend }}_forward(float *m, int s, float *ret, int rs) {
    auto parameterLoader = getParameterLoader("backend/generated/parameters.bin");
""")
postamble = """
    return 0;
}
#ifdef _cplusplus
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

load = Template("parameterLoader->LoadParameters(params, {{num_params}});\n")
parameter_offset = Template("params+{{offset}}")
dense = Template("dense({{input}}, {{h}}, {{w}}, {{weights}}, {{neurons}}, {{biases}}, {{ret}});\n")


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

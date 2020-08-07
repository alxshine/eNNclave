
#include <cstdlib>
#include "backends.h"

#include "nn.h"
#include "IParameterLoader.h"
#include "output.h"

using namespace eNNclave;

#if defined(__cplusplus)
extern "C" {
#endif
int sgx_enclave_forward(float *m, int s, float *ret, int rs) {
    auto parameterLoader = getParameterLoader("backend/generated/parameters.bin.aes");
    print_out("Hello, this is backend sgx\n");
    return 0;
}
#if defined(__cplusplus)
}
#endif

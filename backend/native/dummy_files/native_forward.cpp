
#include <cstdlib>
#include "backends.h"

#include "nn.h"
#include "IParameterLoader.h"
#include "output.h"

using namespace eNNclave;

#ifdef _cplusplus
extern "C" {
#endif
int native_forward(float *m, int s, float *ret, int rs) {
    auto parameterLoader = getParameterLoader("");
    print_out("Hello, this is backend native\n");
    return 0;
}
#ifdef _cplusplus
}
#endif

#include "backends.h"
#include "EnclaveHandler.h"
#include <iostream>


#if defined(__cplusplus)
extern "C" {
#endif

using namespace eNNclave;

int sgx_forward(float *input, int size, float *ret, int returnSize){
    std::cout << "this is the wrapper" << std::endl;
    EnclaveHandler enclaveHandler;
    enclaveHandler.forward(input, size, ret, returnSize);
}

#if defined(__cplusplus)
}
#endif
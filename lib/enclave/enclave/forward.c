#include "forward.h"
#include "matutil.hpp"

int forward(float *m, int size, int *label){
    print("This is the forward function inside the enclave\n");
    matutil_dump_matrix(m, 1, size);
    return 0;
}

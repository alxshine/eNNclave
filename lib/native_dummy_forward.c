#include <stdio.h>

#include "forward.h"

int native_f(float *m, int size, float *ret, int rs){
    print_out("Native dummy forward\n");
    return 0;
}

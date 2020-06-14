#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matutil.h"
#include "enclave_nn.h"

#include "test/util.h"
#include "test/assert.h"
#include "test/multiply.h"
#include "test/add.h"
#include "test/sep_conv1.h"

int main(void)
{
    printf("Testing matutil using native library\n");
    print_separator();

    int correct_cases = 0;
    int total_cases = 0;

    test_multiply(&correct_cases, &total_cases);
    print_separator();
    test_add(&correct_cases, &total_cases);
    print_separator();

    // sep_conv1
    test_sep_conv1(&correct_cases, &total_cases);

    // conv2
    // identity
    // zeros
    // random 1
    // random 2
    // random 3

    // depthwise_conv2
    // identity
    // zeros
    // random 1
    // random 2
    // random 3

    // relu
    // random 1
    // random 2
    // random 3

    // global_average_pooling_1d
    // identity
    // zeros
    // random 1
    // random 2
    // random 3

    // global_average_pooling_2d
    // identity
    // zeros
    // random 1
    // random 2
    // random 3

    // max_pooling_1d
    // identity
    // zeros
    // random 1
    // random 2
    // random 3

    // max_pooling_2d
    // identity
    // zeros
    // random 1
    // random 2
    // random 3

    // zero_pad2
    // random 1
    // random 2

    print_separator();
    printf("%d/%d tests correct\n", correct_cases, total_cases);
    if (correct_cases == total_cases)
    {
        print_separator();
        printf("\n\tAll tests successful :)\n\n");
    }
    print_separator();
    return !(correct_cases == total_cases);
}

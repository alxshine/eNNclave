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
#include "test/conv2.h"
#include "test/relu.h"
#include "test/global_average_pooling1.h"
#include "test/global_average_pooling2.h"
#include "test/max_pool1.h"
#include "test/max_pool2.h"

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
    test_sep_conv1(&correct_cases, &total_cases);

    print_separator();
    test_conv2(&correct_cases, &total_cases);

    print_separator();
    test_relu(&correct_cases, &total_cases);

    print_separator();
    test_global_average_pooling1(&correct_cases, &total_cases);

    print_separator();
    test_global_average_pooling2(&correct_cases, &total_cases);

    print_separator();
    test_max_pool1(&correct_cases, &total_cases);

    print_separator();
    test_max_pool2(&correct_cases, &total_cases);

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

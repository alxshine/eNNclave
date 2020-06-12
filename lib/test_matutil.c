#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matutil.h"
#include "enclave_nn.h"

#include "test/util.h"
#include "test/assert.h"
#include "test/multiply.h"
#include "test/add.h"

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
    // identity
    // zeros
    // random 1
    int steps = 3;
    int channels = 3;
    int filters = 3;
    int kernel_size = 2;

    float inputs[] = {0.473, 0.235, 0.686, 0.159, 0.134, 0.454, 0.16, 0.874, 0.743};
    float depth_kernels[] = {0.437, -0.0364, 0.351, -0.4, -0.792, -0.0659};
    float point_kernels[] = {0.663, 0.552, 0.605, 0.702, -0.792, 0.432, 0.475, -0.825, 0.817};
    float biases[] = {0.0, 0.0, 0.0};
    float expected[] = {0.114, -0.00393, 0.209, -0.434, 0.465, -0.208};
    float ret[steps * filters];
    matutil_sep_conv1(inputs, steps, channels, filters, depth_kernels, point_kernels, kernel_size, biases, ret);
    correct_cases += print_result("Sep conv random 1", assert_similarity(ret, expected, steps * filters));
    total_cases++;

    // random 2
    // random 3

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

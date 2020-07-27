#include "tests.h"
#include "nn.h"

int sep_conv1_zeros() {
    int steps = 3;
    int channels = 2;
    int filters = 3;
    int kernel_size = 3;

    float inputs[] = {0.309, 0.493, 0.711, 0.884, 0.384, 0.389};
    float depth_kernels[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float point_kernels[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float biases[] = {0.0, 0.0, 0.0};
    float expected[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    float ret[steps * filters];
    matutil_sep_conv1(inputs, steps, channels, filters, depth_kernels, point_kernels, kernel_size, biases, ret);

    return print_result("Sep-Conv-1 zeros", assert_similarity(ret, expected, steps * filters));
}

int sep_conv1_random_1() {
    int steps = 3;
    int channels = 3;
    int filters = 3;
    int kernel_size = 3;

    float inputs[] = {0.33, 0.32, 0.171, 0.32, 0.406, 0.986, 0.803, 0.42, 0.398};
    float depth_kernels[] = {0.00116, -0.0863, -0.0745, -0.284, 0.516, 0.537, -0.00431, -0.244, -0.0704};
    float point_kernels[] = {0.962, 0.0922, -0.651, -0.048, -0.705, 0.702, 0.351, -0.891, -0.72};
    float biases[] = {0.761, 0.483, -0.71};
    float expected[] = {0.675, 0.408, -0.619, 0.839, -0.0172, -0.945, 0.583, 0.209, -0.536};

    float ret[steps * filters];
    matutil_sep_conv1(inputs, steps, channels, filters, depth_kernels, point_kernels, kernel_size, biases, ret);
    return print_result("Sep-Conv-1 random 1", assert_similarity(ret, expected, steps * filters));
}

int sep_conv1_random_2() {
    int steps = 3;
    int channels = 3;
    int filters = 3;
    int kernel_size = 3;

    float inputs[] = {0.495, 0.62, 0.971, 0.277, 0.0939, 0.687, 0.625, 0.0469, 0.506};
    float depth_kernels[] = {-0.033, -0.511, -0.6, -0.508, 0.14, 0.312, -0.0767, -0.376, 0.37};
    float point_kernels[] = {-0.165, 0.344, -0.0489, -0.431, -0.636, -0.933, 0.235, -0.306, 0.462};
    float biases[] = {-0.57, -0.419, -0.287};
    float expected[] = {-0.417, -0.716, -0.0644, -0.441, -0.23, -0.0613, -0.558, -0.427, -0.35};

    float ret[steps * filters];
    matutil_sep_conv1(inputs, steps, channels, filters, depth_kernels, point_kernels, kernel_size, biases, ret);
    return print_result("Sep-Conv-1 random 2", assert_similarity(ret, expected, steps * filters));
}

int sep_conv1_random_3() {
    int steps = 10;
    int channels = 4;
    int filters = 5;
    int kernel_size = 3;

    float inputs[] = {0.634, 0.714, 0.858, 0.419, 0.475, 0.637, 0.93, 0.532, 0.015, 0.0276, 0.153, 0.735, 0.527, 0.93,
                      0.488, 0.653, 0.615, 0.962, 0.83, 0.23, 0.386, 0.423, 0.0723, 0.731, 0.921, 0.566, 0.402, 0.273,
                      0.74, 0.466, 0.0985, 0.731, 0.719, 0.841, 0.753, 0.325, 0.633, 0.446, 0.95, 0.724};
    float depth_kernels[] = {0.0509, 0.43, 0.306, -0.572, 0.0624, 0.356, 0.604, 0.47, -0.328, -0.455, 0.333, 0.209};
    float point_kernels[] = {0.306, -0.648, 0.329, -0.129, 0.609, 0.0934, -0.667, -0.398, 0.244, -0.611, 0.782, -0.394,
                             -0.66, 0.0447, 0.556, 0.746, 0.567, -0.63, -0.51, 0.689};
    float biases[] = {-0.646, 0.744, -0.119, -0.649, -0.607};
    float expected[] = {0.192, 0.692, -0.883, -0.763, 0.0166, 0.226, 0.107, -0.988, -0.574, -0.291, -0.15, 0.82, -0.58,
                        -0.73, -0.19, -0.272, 0.635, -0.503, -0.59, -0.353, -0.17, 0.0872, -0.732, -0.42, -0.683,
                        -0.155, 0.682, -0.776, -0.66, -0.52, -0.55, 0.542, -0.352, -0.503, -0.742, -0.158, 0.792,
                        -0.623, -0.735, -0.294, -0.115, 0.246, -0.735, -0.467, -0.498, 0.17, 0.118, -0.929, -0.575,
                        -0.325};

    float ret[steps * filters];
    matutil_sep_conv1(inputs, steps, channels, filters, depth_kernels, point_kernels, kernel_size, biases, ret);
    return print_result("Sep-Conv-1 random 3", assert_similarity(ret, expected, steps * filters));
}

void test_sep_conv1(int* correct_cases, int* total_cases) {
    *correct_cases += sep_conv1_zeros();
    *total_cases += 1;

    *correct_cases += sep_conv1_random_1();
    *total_cases += 1;
    *correct_cases += sep_conv1_random_2();
    *total_cases += 1;
    *correct_cases += sep_conv1_random_3();
    *total_cases += 1;
}
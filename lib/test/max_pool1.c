#include "test/max_pool1.h"

#include "test/util.h"
#include "test/assert.h"
#include "matutil.h"

int max_pool1_random_1()
{
    int steps = 10;
    int channels = 3;
    int pool_size = 3;

    float inputs[] = {-0.984, 0.841, 0.688, -0.61, 0.917, -0.0399, 0.538, 0.0615, 0.859, -0.75, 0.171, 0.923, -0.547, 0.119, -0.793, 0.0482, -0.459, 0.941, -0.79, -0.328, 0.189, -0.123, -0.436, -0.502, 0.901, 0.583, 0.996, -0.802, 0.829, -0.0105};
    float expected[] = {-0.61, 0.917, 0.688, 0.538, 0.171, 0.923, 0.0482, -0.328, 0.941, 0.901, 0.829, 0.996};

    float ret[steps * channels];
    matutil_max_pooling_1d(inputs, steps, channels, pool_size, ret);

    return print_result("MaxPool1D random 1", assert_similarity(expected, ret, steps * channels));
}

int max_pool1_random_2()
{
    int steps = 10;
    int channels = 3;
    int pool_size = 3;

    float inputs[] = {0.293, -0.427, -0.858, 0.0556, 0.782, 0.388, 0.131, 0.323, -0.338, 0.75, -0.352, -0.538, 0.577, 0.593, -0.933, -0.799, 0.00261, -0.974, -0.16, 0.554, 0.788, 0.49, 0.159, -0.499, -0.0799, 0.259, 0.726, -0.679, -0.165, -0.419};
    float expected[] = {0.293, 0.782, 0.388, 0.75, 0.593, -0.338, 0.49, 0.554, 0.788, -0.0799, 0.259, 0.726};

    float ret[steps * channels];
    matutil_max_pooling_1d(inputs, steps, channels, pool_size, ret);

    return print_result("MaxPool1D random 2", assert_similarity(expected, ret, steps * channels));
}

int max_pool1_random_3()
{
    int steps = 20;
    int channels = 5;
    int pool_size = 5;

    float inputs[] = {0.216, -0.618, 0.685, -0.533, 0.377, 0.824, -0.0668, 0.926, -0.298, -0.325, 0.084, 0.155, -0.308, 0.746, 0.525, 0.253, 0.446, -0.499, -0.426, 0.726, 0.553, -0.33, -0.519, -0.131, -0.148, -0.256, -0.977, 0.474, 0.605, -0.622, 0.899, 0.384, 0.456, -0.266, 0.146, 0.148, 0.944, 0.853, -0.258, -0.0307, -0.109, 0.146, 0.0125, -0.488, -0.403, -0.193, -0.729, -0.419, 0.512, 0.519, -0.311, 0.523, -0.155, 0.501, -0.2, -0.125, 0.899, 0.816, 0.836, 0.848, -0.0143, 0.0432, 0.684, 0.756, -0.342, 0.475, -0.534, -0.0838, 0.226, 0.904, -0.94, -0.595, -0.364, -0.13, -0.947, 0.689, 0.0137, -0.23, -0.22, -0.0387, -0.217, 0.984, -0.778, -0.109, -0.245, -0.878, 0.181, -0.774, -0.0306, 0.571, 0.148, 0.386, 0.706, 0.0987, -0.00497, -0.138, 0.31, -0.727, -0.873, 0.653};
    float expected[] = {0.824, 0.446, 0.926, 0.746, 0.726, 0.899, 0.944, 0.853, 0.605, 0.519, 0.475, 0.899, 0.816, 0.836, 0.904, 0.689, 0.984, 0.706, 0.0987, 0.653};

    float ret[steps * channels];
    matutil_max_pooling_1d(inputs, steps, channels, pool_size, ret);

    return print_result("MaxPool1D random 3", assert_similarity(expected, ret, steps * channels));
}

void test_max_pool1(int *correct_cases, int *total_cases)
{
    *correct_cases += max_pool1_random_1();
    *total_cases += 1;

    *correct_cases += max_pool1_random_2();
    *total_cases += 1;

    *correct_cases += max_pool1_random_3();
    *total_cases += 1;
}
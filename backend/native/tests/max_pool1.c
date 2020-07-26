

#include "tests.h"
#include "nn.h"

int max_pool1_sequential()
{
    int steps = 10;
    int channels = 3;
    int pool_size = 3;

    float inputs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0};
    float expected[] = {6.0, 7.0, 8.0, 15.0, 16.0, 17.0, 24.0, 25.0, 26.0};
    float ret[steps * channels];
    matutil_max_pooling_1d(inputs, steps, channels, pool_size, ret);

    int num_poolings = steps / pool_size;

    return print_result("MaxPool1D sequential", assert_similarity(expected, ret, num_poolings * channels));
}

int max_pool1_random_1()
{
    int steps = 10;
    int channels = 3;
    int pool_size = 3;

    float inputs[] = {0.813, -0.47, 0.121, 0.955, -0.0757, 0.463, -0.612, -0.469, 0.0173, 0.66, 0.816, 0.3, -0.188, 0.00158, -0.0351, -0.49, 0.884, -0.428, 0.0768, -0.264, 0.283, -0.621, -0.717, 0.382, 0.771, -0.567, 0.151, 0.606, -0.31, -0.425};
    float expected[] = {0.955, -0.0757, 0.463, 0.66, 0.884, 0.3, 0.771, -0.264, 0.382};
    float ret[steps * channels];
    matutil_max_pooling_1d(inputs, steps, channels, pool_size, ret);

    int num_poolings = steps / pool_size;

    return print_result("MaxPool1D random 1", assert_similarity(expected, ret, num_poolings * channels));
}

int max_pool1_random_2()
{
    int steps = 10;
    int channels = 3;
    int pool_size = 3;

    float inputs[] = {0.813, -0.47, 0.121, 0.955, -0.0757, 0.463, -0.612, -0.469, 0.0173, 0.66, 0.816, 0.3, -0.188, 0.00158, -0.0351, -0.49, 0.884, -0.428, 0.0768, -0.264, 0.283, -0.621, -0.717, 0.382, 0.771, -0.567, 0.151, 0.606, -0.31, -0.425};
    float expected[] = {0.955, -0.0757, 0.463, 0.66, 0.884, 0.3, 0.771, -0.264, 0.382};
    float ret[steps * channels];
    matutil_max_pooling_1d(inputs, steps, channels, pool_size, ret);

    int num_poolings = steps / pool_size;

    return print_result("MaxPool1D random 2", assert_similarity(expected, ret, num_poolings * channels));
}

int max_pool1_random_3()
{
    int steps = 20;
    int channels = 5;
    int pool_size = 5;

    float inputs[] = {-0.536, -0.506, 0.635, -0.977, 0.816, 0.509, 0.0656, 0.0992, -0.0544, -0.074, -0.18, 0.592, -0.49, -0.825, -0.912, -0.89, 0.184, 0.205, -0.523, 0.0284, -0.881, 0.25, 0.99, -0.102, -0.363, -0.848, -0.141, 0.00682, 0.39, -0.105, -0.83, -0.921, -0.571, -0.788, -0.931, 0.683, 0.629, -0.631, -0.631, -0.562, 0.921, 0.261, 0.524, 0.452, -0.432, -0.0266, -0.753, 0.582, -0.421, 0.489, -0.282, 0.214, -0.808, 0.0124, -0.0759, 0.512, 0.0168, -0.338, 0.601, 0.644, 0.849, -0.676, 0.53, 0.438, 0.704, -0.966, -0.359, -0.617, -0.916, -0.626, -0.968, 0.401, 0.0445, -0.957, 0.171, 0.275, 0.866, -0.649, -0.0807, 0.164, -0.819, -0.0595, 0.51, -0.693, -0.402, -0.774, -0.714, 0.445, -0.976, 0.693, -0.375, -0.355, -0.181, 0.515, 0.587, -0.529, -0.409, -0.912, 0.717, 0.461};
    float expected[] = {0.509, 0.592, 0.99, -0.0544, 0.816, 0.921, 0.629, 0.582, 0.452, 0.489, 0.849, 0.401, 0.53, 0.601, 0.704, 0.275, 0.866, 0.51, 0.717, 0.693};

    float ret[steps * channels];
    matutil_max_pooling_1d(inputs, steps, channels, pool_size, ret);

    int num_poolings = steps / pool_size;

    return print_result("MaxPool1D random 3", assert_similarity(expected, ret, num_poolings * channels));
}

void test_max_pool1(int *correct_cases, int *total_cases)
{
    *correct_cases += max_pool1_sequential();
    *total_cases += 1;

    *correct_cases += max_pool1_random_1();
    *total_cases += 1;

    *correct_cases += max_pool1_random_2();
    *total_cases += 1;

    *correct_cases += max_pool1_random_3();
    *total_cases += 1;
}
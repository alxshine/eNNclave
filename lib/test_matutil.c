#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matutil.h"
#include "enclave_nn.h"

// assert that two arra'arrays of n elements each are equal
int assert_equality(float *a, float *b, int n)
{
  for (int i = 0; i < n; ++i)
    if (a[i] != b[i])
      return 0;

  return 1;
}

int assert_similarity(float *a, float *b, int n)
{
  for (int i = 0; i < n; ++i)
  {
    float diff = a[i] - b[i];
    if (abs(diff) > 1e-6)
      return 0;
  }

  return 1;
}

int print_result(const char *name, int success)
{
  printf("TEST - %s:\t\t%s\n", name, success ? "SUCCESS" : "FAILURE");
  return success;
}

void print_separator()
{
  printf("-------------------------------------------------\n");
}

void test_multiply(int *correct_cases, int *total_cases)
{
  // identity
  float identity_a[] = {1., 2., 3.,
                        4., 5., 6.,
                        7., 8., 9.};
  float identity_b[] = {1., 0., 0.,
                        0., 1., 0.,
                        0., 0., 1.};
  float identity_res[9];
  matutil_multiply(identity_a, 3, 3, identity_b, 3, 3, identity_res);
  *correct_cases += print_result("Multiplication identity", assert_equality(identity_res, identity_a, 9));
  *total_cases += 1;

  // zeros
  float zero_a[] = {1., 2., 3.,
                    4., 5., 6.,
                    7., 8., 9.};
  float zero_b[] = {0., 0., 0.,
                    0., 0., 0.,
                    0., 0., 0.};
  float zero_res[9];
  matutil_multiply(zero_a, 3, 3, zero_b, 3, 3, zero_res);
  *correct_cases += print_result("Multiplication zeroes", assert_equality(zero_res, zero_b, 9));
  *total_cases += 1;

  // random 1
  float rand_1_a[] = {0.966, 0.0838, 0.569,
                      0.401, 0.674, 0.371,
                      0.281, 0.929, 0.884};
  float rand_1_b[] = {0.632, 0.715, 0.462,
                      0.0127, 0.558, 0.559,
                      0.854, 0.378, 0.943};
  float rand_1_exp[] = {0.61, 0.0599, 0.263,
                        0.00508, 0.376, 0.207,
                        0.24, 0.351, 0.834};
  float rand_1_res[9];
  matutil_multiply(rand_1_a, 3, 3, rand_1_b, 3, 3, rand_1_res);
  *correct_cases += print_result("Multiplication random 1", assert_similarity(rand_1_res, rand_1_exp, 9));
  *total_cases += 1;

  // random 2
  float rand_2_a[] = {0.441, -0.189, -0.262,
                      -0.138, 0.172, 0.0518,
                      -0.305, 0.288, -0.472};
  float rand_2_b[] = {-0.0613, -0.0571, -0.251,
                      0.435, 0.085, 0.255,
                      0.275, 0.0869, 0.2};
  float rand_2_exp[] = {-0.0271, 0.0108, 0.0658,
                        -0.0599, 0.0146, 0.0132,
                        -0.0838, 0.025, -0.0946};

  float rand_2_res[9];
  matutil_multiply(rand_2_a, 3, 3, rand_2_b, 3, 3, rand_2_res);
  *correct_cases += print_result("Multiplication random 2", assert_similarity(rand_2_res, rand_2_exp, 9));
  *total_cases += 1;

  // random 3
  float rand_3_a[] = {0.831, 0.00504, 0.91,
                      0.972, 0.753, 0.364,
                      0.705, 0.377, 0.452};
  float rand_3_b[] = {0.785, 0.532, 0.915,
                      0.958, 0.569, 0.687,
                      0.328, 0.248, 0.428};
  float rand_3_exp[] = {0.652, 0.00268, 0.832,
                        0.931, 0.428, 0.25,
                        0.231, 0.0936, 0.194};
  float rand_3_res[9];
  matutil_multiply(rand_3_a, 3, 3, rand_3_b, 3, 3, rand_3_res);
  *correct_cases += print_result("Multiplication random 3", assert_similarity(rand_3_res, rand_3_exp, 9));
  *total_cases += 1;
}

void test_add(int *correct_cases, int *total_cases)
{
  // identity
  float identity_a[] = {1., 2., 3.,
                        4., 5., 6.,
                        7., 8., 9.};
  float identity_b[] = {0., 0., 0.,
                        0., 0., 0.,
                        0., 0., 0.};
  float identity_res[9];
  matutil_add(identity_a, 3, 3, identity_b, 3, 3, identity_res);
  *correct_cases += print_result("Addition identity", assert_equality(identity_res, identity_a, 9));
  *total_cases += 1;

  // random 1
  float rand_1_a[] = {-0.0254, -0.319, 0.219,
                      0.324, 0.39, -0.0138,
                      0.359, 0.134, -0.402};
  float rand_1_b[] = {-0.236, 0.0787, -0.0271,
                      0.392, -0.453, -0.318,
                      0.146, 0.423, -0.388};
  float rand_1_exp[] = {-0.261, -0.241, 0.192,
                        0.716, -0.0625, -0.332,
                        0.506, 0.557, -0.79};

  float rand_1_res[9];
  matutil_add(rand_1_a, 3, 3, rand_1_b, 3, 3, rand_1_res);
  *correct_cases += print_result("Addition random 1", assert_similarity(rand_1_res, rand_1_exp, 9));
  *total_cases += 1;

  // random 2
  float rand_2_a[] = {0.0148, 0.478, 0.273,
                      0.27, -0.269, 0.386,
                      0.371, 0.308, -0.267};
  float rand_2_b[] = {-0.21, 0.296, 0.315,
                      0.404, 0.436, -0.333,
                      -0.221, -0.425, -0.213};
  float rand_2_exp[] = {-0.195, 0.773, 0.588,
                        0.674, 0.167, 0.0523,
                        0.15, -0.117, -0.48};

  float rand_2_res[9];
  matutil_add(rand_2_a, 3, 3, rand_2_b, 3, 3, rand_2_res);
  *correct_cases += print_result("Addition random 2", assert_similarity(rand_2_res, rand_2_exp, 9));
  *total_cases += 1;

  // random 3
  float rand_3_a[] = {-0.479, -0.378, -0.283,
                      -0.131, 0.298, -0.232,
                      0.284, -0.31, 0.403};
  float rand_3_b[] = {0.114, 0.479, 0.0337,
                      0.0228, 0.425, 0.33,
                      0.104, -0.119, -0.189};
  float rand_3_exp[] = {-0.365, 0.101, -0.25,
                        -0.108, 0.722, 0.0983,
                        0.387, -0.429, 0.214};
  float rand_3_res[9];
  matutil_add(rand_3_a, 3, 3, rand_3_b, 3, 3, rand_3_res);
  *correct_cases += print_result("Addition random 3", assert_similarity(rand_3_res, rand_3_exp, 9));
  *total_cases += 1;
}

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
  correct_cases += print_result("Sep conv random 1", assert_similarity(ret, expected, steps*filters));
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

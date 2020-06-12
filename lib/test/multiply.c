#include "test/multiply.h"

#include "matutil.h"
#include "test/util.h"
#include "test/assert.h"

#include <stdlib.h>

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
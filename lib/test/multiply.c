#include "test/multiply.h"

#include "matutil.h"
#include "test/util.h"
#include "test/assert.h"

#include <stdlib.h>

int mul_identity(){
  float a[] = {1., 2., 3.,
                        4., 5., 6.,
                        7., 8., 9.};
  float b[] = {1., 0., 0.,
                        0., 1., 0.,
                        0., 0., 1.};
  float res[9];
  matutil_multiply(a, 3, 3, b, 3, 3, res);
  return print_result("Multiplication identity", assert_equality(res, a, 9));
}

int mul_zeros(){
  float a[] = {1., 2., 3.,
                    4., 5., 6.,
                    7., 8., 9.};
  float b[] = {0., 0., 0.,
                    0., 0., 0.,
                    0., 0., 0.};
  float res[9];
  matutil_multiply(a, 3, 3, b, 3, 3, res);
  return print_result("Multiplication zeroes", assert_equality(res, b, 9));
}

int mul_random_1(){
  float a[] = {0.966, 0.0838, 0.569,
                      0.401, 0.674, 0.371,
                      0.281, 0.929, 0.884};
  float b[] = {0.632, 0.715, 0.462,
                      0.0127, 0.558, 0.559,
                      0.854, 0.378, 0.943};
  float exp[] = {0.61, 0.0599, 0.263,
                        0.00508, 0.376, 0.207,
                        0.24, 0.351, 0.834};
  float res[9];
  matutil_multiply(a, 3, 3, b, 3, 3, res);
  return print_result("Multiplication random 1", assert_similarity(res, exp, 9));
}

int mul_random_2(){
  float a[] = {0.441, -0.189, -0.262,
                      -0.138, 0.172, 0.0518,
                      -0.305, 0.288, -0.472};
  float b[] = {-0.0613, -0.0571, -0.251,
                      0.435, 0.085, 0.255,
                      0.275, 0.0869, 0.2};
  float exp[] = {-0.0271, 0.0108, 0.0658,
                        -0.0599, 0.0146, 0.0132,
                        -0.0838, 0.025, -0.0946};

  float res[9];
  matutil_multiply(a, 3, 3, b, 3, 3, res);
  return print_result("Multiplication random 2", assert_similarity(res, exp, 9));
}

int mul_random_3(){
  float a[] = {0.831, 0.00504, 0.91,
                      0.972, 0.753, 0.364,
                      0.705, 0.377, 0.452};
  float b[] = {0.785, 0.532, 0.915,
                      0.958, 0.569, 0.687,
                      0.328, 0.248, 0.428};
  float exp[] = {0.652, 0.00268, 0.832,
                        0.931, 0.428, 0.25,
                        0.231, 0.0936, 0.194};
  float res[9];
  matutil_multiply(a, 3, 3, b, 3, 3, res);
  return print_result("Multiplication random 3", assert_similarity(res, exp, 9));
}

void test_multiply(int *correct_cases, int *total_cases)
{
  *correct_cases += mul_identity();
  *total_cases += 1;

  *correct_cases += mul_zeros();
  *total_cases += 1;

  *correct_cases += mul_random_1();
  *total_cases += 1;

  *correct_cases += mul_random_2();
  *total_cases += 1;

  *correct_cases += mul_random_3();
  *total_cases += 1;
}
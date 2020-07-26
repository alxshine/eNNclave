#include "nn.h"
#include "tests.h"

int add_identity()
{
  float a[] = {1., 2., 3.,
               4., 5., 6.,
               7., 8., 9.};
  float b[] = {0., 0., 0.,
               0., 0., 0.,
               0., 0., 0.};
  float res[9];
  matutil_add(a, 3, 3, b, 3, 3, res);
  return print_result("Addition identity", assert_equality(res, a, 9));
}

int add_random_1()
{
  float a[] = {-0.0254, -0.319, 0.219,
               0.324, 0.39, -0.0138,
               0.359, 0.134, -0.402};
  float b[] = {-0.236, 0.0787, -0.0271,
               0.392, -0.453, -0.318,
               0.146, 0.423, -0.388};
  float exp[] = {-0.261, -0.241, 0.192,
                 0.716, -0.0625, -0.332,
                 0.506, 0.557, -0.79};

  float res[9];
  matutil_add(a, 3, 3, b, 3, 3, res);
  return print_result("Addition random 1", assert_similarity(res, exp, 9));
}

int add_random_2()
{
  float a[] = {0.0148, 0.478, 0.273,
               0.27, -0.269, 0.386,
               0.371, 0.308, -0.267};
  float b[] = {-0.21, 0.296, 0.315,
               0.404, 0.436, -0.333,
               -0.221, -0.425, -0.213};
  float exp[] = {-0.195, 0.773, 0.588,
                 0.674, 0.167, 0.0523,
                 0.15, -0.117, -0.48};

  float res[9];
  matutil_add(a, 3, 3, b, 3, 3, res);
  return print_result("Addition random 2", assert_similarity(res, exp, 9));
}

int add_random_3()
{
  float a[] = {-0.479, -0.378, -0.283,
               -0.131, 0.298, -0.232,
               0.284, -0.31, 0.403};
  float rand_3_b[] = {0.114, 0.479, 0.0337,
                      0.0228, 0.425, 0.33,
                      0.104, -0.119, -0.189};
  float exp[] = {-0.365, 0.101, -0.25,
                 -0.108, 0.722, 0.0983,
                 0.387, -0.429, 0.214};
  float res[9];
  matutil_add(a, 3, 3, rand_3_b, 3, 3, res);
  return print_result("Addition random 3", assert_similarity(res, exp, 9));
}

void test_add(int *correct_cases, int *total_cases)
{
  *correct_cases += add_identity();
  *total_cases += 1;

  *correct_cases += add_random_1();
  *total_cases += 1;

  *correct_cases += add_random_2();
  *total_cases += 1;
}
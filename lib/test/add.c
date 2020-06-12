#include "test/add.h"

#include "matutil.h"
#include "test/util.h"
#include "test/assert.h"

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
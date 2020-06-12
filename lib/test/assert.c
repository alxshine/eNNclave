#include "test/assert.h"

#include <stdlib.h>

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
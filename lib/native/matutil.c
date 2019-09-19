#include <stdio.h>

#include "matutil.h"
#include "state.h"

int matutil_initialize(void) { return 0; }

int matutil_teardown(void) { return 0; }

void matutil_get_new_dimensions(int r1, int c1, int r2, int c2, int *rr,
                                int *cr) {
  *rr = r1;
  *cr = c2;
}

int matutil_multiply(float *m1, int r1, int c1, float *m2, int r2, int c2,
                     float *ret) {
  // check dimensions
  if (c1 != r2) {
    fprintf(stderr,
            "Matrices have incompatible dimensions for multiplication %dx%d "
            "and %dx%d\n",
            r1, c1, r2, c2);
    return -1;
  }

  int rr = r1, cr = c2;
  for (int y = 0; y < rr; ++y) { // coordinates in ret
    for (int x = 0; x < cr; ++x) {
      ret[y * cr + x] = 0;
      for (int i = 0, j = 0; i < c1; ++i, ++j) {
        ret[y * cr + x] += m1[y * c1 + j] * m2[i * c2 + x];
      }
    }
  }
  return 0;
}

int matutil_add(float *m1, int r1, int c1, float *m2, int r2, int c2,
                float *ret) {
  if (r1 != r2 || c1 != c2) {
    fprintf(
        stderr,
        "Matrices have incompatible dimensions for addition %dx%d and %dx%d\n",
        r1, c1, r2, c2);
    return -1;
  }

  for (int i = 0; i < r1; ++i) {
    for (int j = 0; j < c1; ++j) {
      int coord = i * c1 + j;
      ret[coord] = m1[coord] + m2[coord];
    }
  }
  return 0;
}

void matutil_relu(float *m, int r, int c) {
  for (int i = 0; i < r * c; i++)
    if (m[i] < 0)
      m[i] = 0;
}

int matutil_dense(float *m, int r, int c, int *label) {
  if (r != 1 || c != w1_r) {
    fprintf(stderr, "Input should be 1x%d\n", w1_r);
    return -1;
  }
  int sts;

  // fc1
  float tmp1[w1_c];
  if ((sts = matutil_multiply(m, r, c, w1, w1_r, w1_c, tmp1)))
    return sts;
  if ((sts = matutil_add(tmp1, 1, w1_c, b1, 1, b1_c, tmp1)))
    return sts;
  matutil_relu(tmp1, 1, w1_c);

  // fc1
  float tmp2[w2_c];
  if ((sts = matutil_multiply(tmp1, 1, w1_c, w2, w2_r, w2_c, tmp2)))
    return sts;
  if ((sts = matutil_add(tmp2, 1, w2_c, b2, 1, b2_c, tmp2)))
    return sts;
  matutil_relu(tmp2, 1, w2_c);

  // get maximum for label
  int max_index = 0;
  for (int i = 1; i < w2_c; ++i) 
    max_index = tmp2[i] > tmp2[max_index] ? i : max_index;

  *label = max_index;
  return 0;
}

void matutil_dump_matrix(float *m, int r, int c) {
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      printf("%f, ", m[i * c + j]);
    }
    printf("\n");
  }
}

#include "matmul.h"

void matmul_get_new_dimensions(int w1, int h1, int w2, int h2, int *wr,
                               int *hr) {
  *wr = w2;
  *hr = h1;
}

int matmul_multiply(float *m1, int w1, int h1, float *m2, int w2, int h2,
                    float *ret) {
  // check dimensions
  if (w1 != h2) {
    fprintf(stderr, "Matrices have incompatible dimensions %dx%d and %dx%d\n",
            w1, h1, w2, h2);
    return -1;
  }
  
  int wr = w2, hr = h1;
  for (int y = 0; y < hr; ++y) { // coordinates in ret
    for (int x = 0; x < wr; ++x) {
      ret[y * wr + x] = 0;
      for (int i = 0, j = 0; i < w1; ++i, ++j) {
        ret[y * wr + x] += m1[y * w1 + j] * m2[i * w2 + x];
      }
    }
  }
  return 0;
}

void matmul_dump_matrix(float *m, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      printf("%f, ", m[i * w + j]);
    }
    printf("\n");
  }
}

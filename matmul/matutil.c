#include "matutil.h"

void matutil_get_new_dimensions(int w1, int h1, int w2, int h2, int *wr,
                               int *hr) {
  *wr = w2;
  *hr = h1;
}

int matutil_multiply(float *m1, int w1, int h1, float *m2, int w2, int h2,
                    float *ret) {
  // check dimensions
  if (w1 != h2) {
    fprintf(stderr, "Matrices have incompatible dimensions for multiplication %dx%d and %dx%d\n",
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

int matutil_add(float *m1, int w1, int h1, float *m2, int w2, int h2, float *ret){
  if(w1 != w2 || w2 != h2){
    fprintf(stderr, "Matrices have incompatible dimensions for addition %dx%d and %dx%d\n",
            w1, h1, w2, h2);
    return -1;
  }

  for(int i=0; i<h1; ++i){
    for (int j=0; j < w1; ++j) {
      int coord = i*w1+j;
      ret[coord] = m1[coord] + m2[coord];
    }
  }
  return 0;
}

void matutil_relu(float *m, int w, int h){
  for(int i=0; i<w*h; i++)
    if(m[i]<0)
      m[i] = 0;
}

void matutil_dump_matrix(float *m, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      printf("%f, ", m[i * w + j]);
    }
    printf("\n");
  }
}

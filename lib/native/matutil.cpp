#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "matutil.hpp"
#include "state.hpp"
#include "dense.hpp"

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

void matutil_dump_matrix(float *m, int r, int c) {
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      printf("%f, ", m[i * c + j]);
    }
    printf("\n");
  }
}

int matutil_dense(float *m, int r, int c, int *label){
  return dense(m, r*c, r, c, label);
}

int print_error(const char* fmt, ...){
  char buf[BUFSIZ] = { '\0' };
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, BUFSIZ, fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s", buf);
  return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

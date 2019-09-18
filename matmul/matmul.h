#ifndef MATMUL_H
#define MATMUL_H

#include <stdio.h>

void matmul_get_new_dimensions(int w1, int h1, int w2, int h2, int *wr, int *hr);

void matmul_multiply(float *m1, int w1, int h1, float *m2, int w2, int h2, float *ret);

void matmul_dump_matrix(float *m, int w, int h);

#endif /* MATMUL_H */

#ifndef MATUTIL_H
#define MATUTIL_H

#include <stdio.h>

#if defined(__cplusplus)
extern "C" {
#endif

int matutil_initialize();

int matutil_teardown();

void matutil_get_new_dimensions(int r1, int c1, int r2, int c2, int *rr, int *cr);

int matutil_multiply(float *m1, int r1, int c1, float *m2, int r2, int c2, float *ret);

int matutil_add(float *m1, int r1, int c1, float *m2, int r2, int c2, float *ret);

void matutil_relu(float *m, int r, int c);

int matutil_dense(float *m, int r, int c, int *label);

void matutil_dump_matrix(float *m, int r, int c);

#if defined(__cplusplus)
}
#endif

#endif /* MATUTIL_H */
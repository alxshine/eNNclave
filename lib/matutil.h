#ifndef MATUTIL_H
#define MATUTIL_H

#include <stdio.h>

int matutil_initialize();

int matutil_teardown();

void matutil_get_new_dimensions(int w1, int h1, int w2, int h2, int *wr, int *hr);

int matutil_multiply(float *m1, int w1, int h1, float *m2, int w2, int h2, float *ret);

int matutil_add(float *m1, int w1, int h1, float *m2, int w2, int h2, float *ret);

void matutil_relu(float *m, int w, int h);

void matutil_dump_matrix(float *m, int w, int h);

#endif /* MATUTIL_H */

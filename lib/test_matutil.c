#include <stdio.h>

#include "matutil.hpp"

int main(void) {
  matutil_initialize();
  printf("Matutil initialized\n");

  int r1 = 2, c1 = 3;
  float m1[] = {1, 2, 3, 4, 5, 6};
  int r2 = 3, c2 = 2;
  float m2[] = {7, 8, 9, 10, 11, 12};

  printf("m1:\n");
  matutil_dump_matrix(m1, r1, c1);
  printf("m2:\n");
  matutil_dump_matrix(m2, r2, c2);

  int rret, cret;
  printf("get_new_dimensions\n");
  matutil_get_new_dimensions(r1, c1, r2, c2, &rret, &cret);
  float mret[rret * cret];
  printf("multiply\n");
  if (matutil_multiply(m1, r1, c1, m2, r2, c2, mret)) {
    return 1;
  };

  printf("mret:\n");
  matutil_dump_matrix(mret, rret, cret);

  float mret2[rret * cret];
  printf("add\n");
  if (matutil_add(mret, rret, cret, mret, rret, cret, mret2))
    return 1;

  printf("mret2:\n");
  matutil_dump_matrix(mret2, rret, cret);

  mret2[2] = -1;
  printf("relu\n");
  matutil_relu(mret2, rret, cret);

  printf("mret2:\n");
  matutil_dump_matrix(mret2, rret, cret);
  
  int c = 9216;
  float input[c];
  for (int i = 0; i<c; ++i) {
    input[i] = i;
  }

  int label = -1;
  printf("forward\n");
  matutil_forward(input, 1, c, &label);
  printf("Output label: %d\n", label);
  
  matutil_teardown();
  printf("Matutil torn down\n");
  return 0;
}

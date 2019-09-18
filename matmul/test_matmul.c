#include <stdio.h>

#include "matmul.h"

int main(void) {
  int w1 = 3, h1 = 2;
  float m1[] = {1, 2, 3, 4, 5, 6};
  int w2 = 2, h2 = 3;
  float m2[] = {7, 8, 9, 10, 11, 12};

  printf("m1:\n");
  matmul_dump_matrix(m1, w1, h1);
  printf("m2:\n");
  matmul_dump_matrix(m2, w2, h2);

  int wret, hret;
  matmul_get_new_dimensions(w1,h1,w2,h2, &wret, &hret);
  float mret[wret*hret];
  if(matmul_multiply(m1, w1-1, h1, m2, w2, h2, mret)){
    return 1;
  };

  printf("mret:\n");
  matmul_dump_matrix(mret, wret, hret);

  return 0;
}

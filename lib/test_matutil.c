#include <stdio.h>
#include <math.h>

/* #include "matutil.hpp" */
#include "native_nn.h"
#include "enclave_nn.h"

int main(void) {
  // matutil_initialize();
  enclave_nn_start();
  printf("Matutil initialized\n");

  /* int r1 = 2, c1 = 3; */
  /* float m1[] = {1, 2, 3, 4, 5, 6}; */
  /* int r2 = 3, c2 = 2; */
  /* float m2[] = {7, 8, 9, 10, 11, 12}; */

  /* printf("m1:\n"); */
  /* matutil_dump_matrix(m1, r1, c1); */
  /* printf("m2:\n"); */
  /* matutil_dump_matrix(m2, r2, c2); */

  /* int rret, cret; */
  /* printf("get_new_dimensions\n"); */
  /* matutil_get_new_dimensions(r1, c1, r2, c2, &rret, &cret); */
  /* float mret[rret * cret]; */
  /* printf("multiply\n"); */
  /* if (matutil_multiply(m1, r1, c1, m2, r2, c2, mret)) { */
  /*   return 1; */
  /* }; */

  /* printf("mret:\n"); */
  /* matutil_dump_matrix(mret, rret, cret); */

  /* float mret2[rret * cret]; */
  /* printf("add\n"); */
  /* if (matutil_add(mret, rret, cret, mret, rret, cret, mret2)) */
  /*   return 1; */

  /* printf("mret2:\n"); */
  /* matutil_dump_matrix(mret2, rret, cret); */

  /* mret2[2] = -1; */
  /* printf("relu\n"); */
  /* matutil_relu(mret2, rret, cret); */

  /* printf("mret2:\n"); */
  /* matutil_dump_matrix(mret2, rret, cret); */

  int c = 14*14*512;
  float input[c];
  for (int i = 0; i < c; ++i) {
    input[i] = i;
  }

  int native_label = -1;
  int enclave_label = -1;
  enclave_nn_forward(input, c, &enclave_label);
  native_nn_forward(input, c, &native_label);
  printf("Enclave label: %d\n", enclave_label);
  printf("Native label: %d\n", native_label);

  /* printf("\n\nConvolution:\n"); */
  /* int conv_input_size = 18; */
  /* float conv_input[conv_input_size]; */
  /* for (int i = 0; i < conv_input_size; ++i) { */
  /*   conv_input[i] = i; */
  /* } */
  /* float weights[36]; */
  /* for (int i = 0; i < 36; ++i) { */
  /*   weights[i] = 1; */
  /* } */

  /* float biases[2] = {0,0}; */

  /* float results[18]; */
  /* matutil_conv2(conv_input, 3, 3, 2, 2, weights, 3, 3, biases, results); */

  /* printf("Inputs:\n"); */
  /* matutil_dump_matrix3(conv_input, 3, 3, 2); */
  /* printf("Weights:\n"); */
  /* matutil_dump_matrix3(weights, 3, 3, 4); */
  /* printf("Outputs:\n"); */
  /* matutil_dump_matrix3(results, 3, 3, 2); */

  /* printf("\n\nGlobalAveragePooling2D:\n"); */
  /* int gap_input_size = 18; */
  /* float gap_input[gap_input_size]; */
  /* for (int i = 0; i < gap_input_size; ++i) { */
  /*   gap_input[i] = i; */
  /* } */

  /* float gap_results[2]; */
  /* matutil_global_average_pooling_2d(gap_input, 3, 3, 2, gap_results); */

  /* printf("Inputs:\n"); */
  /* matutil_dump_matrix3(gap_input, 3, 3, 2); */
  /* printf("Outputs:\n"); */
  /* matutil_dump_matrix(gap_results, 1, 2); */

  /* printf("\n\nMaxPooling2D:\n"); */
  /* int mp_input_size = 32; */
  /* float mp_input[mp_input_size]; */
  /* for (int i=0; i<mp_input_size; ++i) { */
  /*   mp_input[i] = i; */
  /* } */

  /* float mp_results[mp_input_size/4]; */
  /* matutil_max_pooling_2d(mp_input, 4, 4, 2, 2, mp_results); */

  /* printf("Inputs:\n"); */
  /* matutil_dump_matrix3(mp_input, 4,4,2); */
  /* printf("Outputs:\n"); */
  /* matutil_dump_matrix3(mp_results, 2,2,2); */

  // /* enclave_teardown(); */
  /* matutil_teardown(); */
  printf("Matutil torn down\n");
  return 0;
}

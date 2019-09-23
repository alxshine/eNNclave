/*
 * Copyright (C) 2011-2019 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>

/* 
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

void do_something(void){
    printf("If you can see this, the enclave works\n");
}

int multiply(int a, int b){
    return a*b;
}

void dump_matrix(float *m, int s, int, int c){
  for (int i = 0; i<s; ++i) {
    printf("%f, ", m[i]);
    if(i%c==c-1)
      printf("\n");
  }

}

int matmul(float *m1, int s1, int r1, int c1, float *m2, int s2, int r2, int c2,
	   float *ret, int sr) {

  // check dimensions
  if (c1 != r2)
    return 1;

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

int add(float *m1, int s1, int r1, int c1, float *m2, int s2, int r2, int c2,
	float *ret, int sr) {
  if (r1 != r2 || c1 != c2) 
    return 2;
  
  for (int i = 0; i < r1; ++i) {
    for (int j = 0; j < c1; ++j) {
      int coord = i * c1 + j;
      ret[coord] = m1[coord] + m2[coord];
    }
  }
  return 0;
}

void relu(float *m, int s, int r, int c) {
  for (int i = 0; i < r * c; i++)
    if (m[i] < 0)
      m[i] = 0;
}

int dense(float *m, int r, int c, int *label) {
//   if (r != 1 || c != w1_r)
//     return 3;
//   int sts;

//   // fc1
//   float tmp1[w1_c];
//   if ((sts = matmul(m, r, c, w1, w1_r, w1_c, tmp1)))
//     return sts;
//   if ((sts = add(tmp1, 1, w1_c, b1, 1, b1_c, tmp1)))
//     return sts;
//   relu(tmp1, 1, w1_c);

//   // fc1
//   float tmp2[w2_c];
//   if ((sts = matmul(tmp1, 1, w1_c, w2, w2_r, w2_c, tmp2)))
//     return sts;
//   if ((sts = add(tmp2, 1, w2_c, b2, 1, b2_c, tmp2)))
//     return sts;
//   relu(tmp2, 1, w2_c);

//   // get maximum for label
//   int max_index = 0;
//   for (int i = 1; i < w2_c; ++i) 
//     max_index = tmp2[i] > tmp2[max_index] ? i : max_index;

//   *label = max_index;
  return 0;
}

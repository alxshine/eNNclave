#include "enclave.h"
#include "enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>

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

int print_error(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

void test(void){
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

#include <stdio.h>

#include "matutil.h"

int main(void) {
  matutil_initialize();

  float input[800];
  for (int i = 0; i<800; ++i) {
    input[i] = i;
  }

  int label;
  matutil_dense(input, 1, 800, &label);
  printf("Output label: %d, should be 5\n", label);
  
  matutil_teardown();
  return 0;
}

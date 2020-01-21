#include "forward.h"
#include "native_nn.h"

int native_forward(float *m, int size, int *label){
  return forward(m, size, label);
}

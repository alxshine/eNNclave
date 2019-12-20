#include "forward.hpp"
#include "native.hpp"

int native_forward(float *m, int size, int *label){
  return forward(m, size, label);
}

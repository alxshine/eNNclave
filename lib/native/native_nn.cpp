#include "forward.hpp"
#include "native_nn.hpp"

int native_forward(float *m, int size, int *label){
  return forward(m, size, label);
}

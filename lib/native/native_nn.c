#include "forward.h"
#include "native_nn.h"

int native_nn_forward(float *m, int size, float *ret, int rs){
  return native_f(m, size, ret, rs);
}

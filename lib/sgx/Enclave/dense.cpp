
#include "state.hpp"
#include "matutil.hpp"
#include "dense.hpp"

int dense(float *m, int, int r, int c, int *label) {
  if (r != 1 || c != 9216) {
    print_error("ERROR: Input should be 1x9216, got %dx%d\n", r, c);
    return -1;
  }
  int sts;

  float tmp0[128];
  if ((sts = matutil_multiply(m, 1, 9216, w0, 9216, 128, tmp0)))
    return sts;
  if ((sts = matutil_add(tmp0, 1, 128, b0, 1, 128, tmp0)))
    return sts;
  matutil_relu(tmp0, 1, 128);

  //No call method generated for layer dropout of type <class 'tensorflow.python.keras.layers.core.Dropout'>

  float tmp2[10];
  if ((sts = matutil_multiply(tmp0, 1, 128, w2, 128, 10, tmp2)))
    return sts;
  if ((sts = matutil_add(tmp2, 1, 10, b2, 1, 10, tmp2)))
    return sts;

  // get maximum for label
  int max_index = 0;
  for (int i = 1; i < 10; ++i)
    max_index = tmp2[i] > tmp2[max_index] ? i : max_index;

  *label = max_index;


  return 0;
}


#include "state.hpp"
#include "matutil.hpp"

int matutil_dense(float *m, int r, int c, int *label) {
  if (r != 1 || c != 10) {
    fprintf(stderr, "ERROR: Input should be 1x10, got %dx%d\n", r, c);
    return -1;
  }
  int sts;

  float tmp0[2];
  if ((sts = matutil_multiply(m, 1, 10, w0, 10, 2, tmp0)))
    return sts;
  if ((sts = matutil_add(tmp0, 1, 2, b0, 1, 2, tmp0)))
    return sts;
  matutil_relu(tmp0, 1, 2);

  float tmp1[1];
  if ((sts = matutil_multiply(tmp0, 1, 2, w1, 2, 1, tmp1)))
    return sts;
  if ((sts = matutil_add(tmp1, 1, 1, b1, 1, 1, tmp1)))
    return sts;

  // get maximum for label
  int max_index = 0;
  for (int i = 1; i < 1; ++i)
    max_index = tmp1[i] > tmp1[max_index] ? i : max_index;

  *label = max_index;


  return 0;
}

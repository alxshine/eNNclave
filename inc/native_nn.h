#ifndef NATIVE_H
#define NATIVE_H

#if defined(__cplusplus)
extern "C" {
#endif

  int native_nn_forward(float *m, int s, int *label);

#if defined(__cplusplus)
}
#endif
    
#endif /* NATIVE_H */

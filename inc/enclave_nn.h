#ifndef ENCLAVE_H
#define ENCLAVE_H

#if defined(__cplusplus)
extern "C" {
#endif

  int enclave_nn_start();
  int enclave_nn_end();
  int enclave_nn_forward(float *m, int s, float *ret, int rs);

#if defined(__cplusplus)
}
#endif
    
#endif /* ENCLAVE_H */

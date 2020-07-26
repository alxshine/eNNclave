#ifndef ENCLAVE_H
#define ENCLAVE_H

#if defined(__cplusplus)
extern "C" {
#endif

  int sgx_start();
  int sgx_end();
  int sgx_forward(float *m, int s, float *ret, int rs);

#if defined(__cplusplus)
}
#endif
    
#endif /* ENCLAVE_H */

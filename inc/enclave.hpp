#ifndef ENCLAVE_H
#define ENCLAVE_H

#if defined(__cplusplus)
extern "C" {
#endif

  int enclave_initialize();
  int enclave_teardown();
  int enclave_forward();

#if defined(__cplusplus)
}
#endif
    
#endif /* ENCLAVE_H */

#ifndef wrapper_h
#define wrapper_h

#include "sgx_urts.h"

extern sgx_enclave_id_t enclave_id;

#if defined(__cplusplus)
extern "C" {
#endif

 void ocall_stdout_string(const char* str);
 void ocall_stderr_string(const char* str);

 int encrypt_parameter_file(const char* path);

#if defined(__cplusplus)
}
#endif

#endif // wrapper_h


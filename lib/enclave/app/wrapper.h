#ifndef app/wrapper_h_INCLUDED
#define app/wrapper_h_INCLUDED

extern sgx_enclave_id_t enclave_id;

#if defined(__cplusplus)
extern "C" {
#endif

 void ocall_stdout_string(const char* str);
 void ocall_stderr_string(const char* str);

#if defined(__cplusplus)
}
#endif

#endif // app/wrapper_h_INCLUDED


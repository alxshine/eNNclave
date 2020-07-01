#ifndef _enclave_h
#define _enclave_h

#include "sgx_tprotected_fs.h"

#if defined(__cplusplus)
extern "C" {
#endif

    extern SGX_FILE *encrypted_parameters;

    void test();

    int print_out(const char* fmt, ...);
    int print_err(const char* fmt, ...);

    void open_encrypted_parameters();
    int encrypt_parameters(float *target_buffer, int num_elements);
    void close_encrypted_parameters();
#if defined(__cplusplus)
}
#endif

#endif

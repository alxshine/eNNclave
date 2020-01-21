#include <stdio.h>

#include "enclave_nn.h"
#include "enclave_u.h"
#include "utils.h"
#include "sgx_urts.h"

sgx_enclave_id_t enclave_id;
const char *enclave_filename = "enclave.signed.so";

void ocall_stdout_string(const char *str){
    printf("%s", str);
}

void ocall_stderr_string(const char *str){
    fprintf(stderr, "%s", str);
}

int enclave_nn_start(){
    sgx_status_t ret = sgx_create_enclave(enclave_filename, SGX_DEBUG_FLAG, NULL, NULL, &enclave_id, NULL);
    if( ret != SGX_SUCCESS){
        print_error_message(ret);
        return 1;
    }
    return 0;
}

int enclave_nn_end(){
    sgx_destroy_enclave(enclave_id);
}

int enclave_nn_forward(float *m, int s, int *label){
    // sgx_status_t ret = test(enclave_id);
    int return_value;
    sgx_status_t ret = forward(enclave_id, &return_value, m, s, label);
    if(ret != SGX_SUCCESS){
        print_error_message(ret);
        return 1;
    }
    return 0;
};

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
    printf("This is the enclave wrapper\n");
    int return_value;
    sgx_status_t ret = enclave_f(enclave_id, &return_value, m, s, label);
    if(ret != SGX_SUCCESS){
        print_error_message(ret);
        return 1;
    }
    return 0;
};

int encrypt_parameter_file(const char* path){
    sgx_status_t ret = open_encrypted_parameters(enclave_id);
    if (ret != SGX_SUCCESS){
        perror("Could not open encrypted parameter file");
        print_error_message(ret);
        return 1;
    }

    FILE *raw_parameter_file = fopen(path, "r");
    if (raw_parameter_file == NULL){
        perror("Could not open parameter file");
        return 1;
    }

    float buffer[1024];
    int num_read;
    do {
        num_read = fread(buffer, sizeof(float), 1024, raw_parameter_file);
        if(num_read == 0)
            break;

        int num_wrote;
        ret = encrypt_parameters(enclave_id, &num_wrote, buffer, num_read);
        if (ret != SGX_SUCCESS){
            print_error_message(ret);
            close_encrypted_parameters(enclave_id);
            return 1;
        }
    } while(num_read = 1024);

    ret = close_encrypted_parameters(enclave_id);
    if (ret != SGX_SUCCESS){
        close_encrypted_parameters(enclave_id);
        return 1;
    }
    return 0;
}

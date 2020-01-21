#include <iostream>

#include "enclave_nn.hpp"
#include "enclave_u.h"
#include "utils.h"
#include "sgx_urts.h"

using namespace std;

sgx_enclave_id_t enclave_id;
const char *enclave_filename = "enclave.signed.so";

int enclave_nn_start(){
    cout << "initializing enclave" << endl;
    sgx_status_t ret = sgx_create_enclave(enclave_filename, SGX_DEBUG_FLAG, NULL, NULL, &enclave_id, NULL);
    if( ret != SGX_SUCCESS){
        print_error_message(ret);
        return 1;
    }
    return 0;
}

int enclave_nn_end(){
    cout << "destroying enclave" << endl;
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

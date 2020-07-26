#include <stdio.h>

#include "app.h"
#include "wrapper.h"
#include "backend_sgx.h"
#include "sgx_urts.h"
#include "enclave_u.h"

int main(void)
{
    // sgx_enclave_id_t enclave_id = 0;
    // const char* enclave_filename = "sgx.signed.so";
    // cout << "Trying to access " << enclave_filename << endl;
    // ifstream ifs{enclave_filename};
    // if(ifs.is_open()) {
    //     cout << "Success" << endl;
    // } else {
    //     cerr << "Could not open file " << enclave_filename << endl;
    //     return 1;
    // }

    // cout << "Trying to create sgx";
    // if (SGX_DEBUG_FLAG)
    //     cout << " in debug mode" << endl;
    // else
    //     cout << " without debug mode" << endl;

    // sgx_status_t ret = sgx_create_enclave("sgx.signed.so", SGX_DEBUG_FLAG, NULL, NULL, &enclave_id, NULL);
    // if( ret != SGX_SUCCESS){
    //     print_error_message(ret);
    //     return 1;
    // }
    if(enclave_nn_start())
        return 1;

    encrypt_parameter_file("../../parameters.bin");

    enclave_nn_end();
    // sgx_destroy_enclave(enclave_id);
    // cout << "Info: Enclave successfully returned." << endl;
    // cout << "Enter a character before exit ...\n" << endl;
    // getchar();
    return 0;

}

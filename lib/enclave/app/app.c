#include <iostream>
#include <fstream>

#include "app.h"
#include "enclave_nn.hpp"
#include "sgx_urts.h"
#include "enclave_u.h"

using namespace std;

int main(void)
{
    // sgx_enclave_id_t enclave_id = 0;
    // const char* enclave_filename = "enclave.signed.so";
    // cout << "Trying to access " << enclave_filename << endl;
    // ifstream ifs{enclave_filename};
    // if(ifs.is_open()) {
    //     cout << "Success" << endl;
    // } else {
    //     cerr << "Could not open file " << enclave_filename << endl;
    //     return 1;
    // }

    // cout << "Trying to create enclave";
    // if (SGX_DEBUG_FLAG)
    //     cout << " in debug mode" << endl;
    // else
    //     cout << " without debug mode" << endl;

    // sgx_status_t ret = sgx_create_enclave("enclave.signed.so", SGX_DEBUG_FLAG, NULL, NULL, &enclave_id, NULL);
    // if( ret != SGX_SUCCESS){
    //     print_error_message(ret);
    //     return 1;
    // }
    if(enclave_nn_start())
        return 1;

    int size=20;
    float m[size];
    for(int i = 0; i<size; ++i)
        m[i] = i;
    int label = -1;

    if(enclave_nn_forward(m, size, &label))
        return 1;

    // sgx_status_t ret = test(enclave_id);
    // if( ret != SGX_SUCCESS){
    //     print_error_message(ret);
    //     return 1;
    // }


    enclave_nn_end();
    // sgx_destroy_enclave(enclave_id);
    // cout << "Info: Enclave successfully returned." << endl;
    // cout << "Enter a character before exit ...\n" << endl;
    // getchar();
    return 0;

}

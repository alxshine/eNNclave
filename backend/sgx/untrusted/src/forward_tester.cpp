#include <iostream>
#include <dlfcn.h>

#include "backends.h"

using namespace std;

int main(void)
{
    cout << "Opening sgx backend library" << endl;
    void *sgx_backend_handle = dlopen("libbackend_sgx.so", RTLD_LAZY);
    if (sgx_backend_handle == nullptr)
    {
        cerr << "Could not open library" << endl;
        return 1;
    }
    cout << "Library successfully opened" << endl;
    dlerror();

    cout << "Finding forward function" << endl;
    FORWARD_T *sgx_forward = (FORWARD_T *)dlsym(sgx_backend_handle, "sgx_forward");
    if (dlerror())
    {
        cerr << "Could not find forward function" << endl;
        return 1;
    }

    auto input_size = 100, output_size = 100;
    float input[input_size], ret[output_size];
    for (size_t i = 0; i < input_size; i++)
        input[i] = 0;
    

    int status = (*sgx_forward)(input, input_size, ret, output_size);
    if(status){
        cerr << "Error during SGX forward" << endl;
        return 1;
    }

    return 0;
}
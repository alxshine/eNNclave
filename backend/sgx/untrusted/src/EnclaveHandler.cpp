#include "EnclaveHandler.h"

#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include "enclave_u.h"

using namespace std;

eNNclave::EnclaveHandler::EnclaveHandler(){
    sgx_status_t status = sgx_create_enclave(EnclaveHandler::enclaveFilename, SGX_DEBUG_FLAG, nullptr, nullptr, &enclaveId, nullptr);
    if(status != SGX_SUCCESS)
        throw logic_error{"could not initialize enclave, error code: " + to_string(status)};
}

eNNclave::EnclaveHandler::~EnclaveHandler() noexcept{
    sgx_destroy_enclave(enclaveId);
}

void eNNclave::EnclaveHandler::forward(float *input, int size, float *ret, int returnSize){
    int returnValue;
    sgx_status_t status = sgx_enclave_forward(enclaveId, &returnValue, input, size, ret, returnSize);
    if (status != SGX_SUCCESS)
        throw logic_error{"sgx_forward returned error code: " + to_string(status)};
}

void eNNclave::EnclaveHandler::encryptParameterFile(const string &parameterFilename){
    ifstream parameterFile{parameterFilename, ifstream::binary};
    if(!parameterFile.good())
        throw logic_error{"Could not open input file " + parameterFilename};

    auto status = open_encrypted_parameters(enclaveId);
    if(status != SGX_SUCCESS)
        throw logic_error{"Could not open encrypted parameter file. Error code: " + to_string(status)};

    float buffer[1024];
    while (true) {
        auto numRead = parameterFile.readsome((char*)buffer, 1024*sizeof(float))/sizeof(float);
        if(numRead == 0)
            break;

        int num_wrote;
        status = encrypt_parameters(enclaveId, &num_wrote, buffer, numRead);
        if (status != SGX_SUCCESS){
            close_encrypted_parameters(enclaveId);
            throw logic_error{"sgx_forward returned error code: " + to_string(status)};
        }
    }

    status = close_encrypted_parameters(enclaveId);
    if(status != SGX_SUCCESS)
        throw logic_error{"Error closing encrpyted parameter file: " + to_string(status)};
}
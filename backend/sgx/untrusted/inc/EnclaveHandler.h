#ifndef ENCLAVEMANAGER_H
#define ENCLAVEMANAGER_H

#include "sgx_urts.h"
#include <string>

namespace eNNclave
{
    class EnclaveHandler
    {
    public:
        EnclaveHandler();
        ~EnclaveHandler() noexcept;

        void forward(float *input, int size, float *ret, int returnSize);
        void encryptParameterFile(const std::string& parameterFilename);
    private:
        sgx_enclave_id_t enclaveId;
    };

}; // namespace eNNclave

#endif
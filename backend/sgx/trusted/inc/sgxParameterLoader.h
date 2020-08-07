#ifndef SGXPARAMETERLOADER_H
#define SGXPARAMETERLOADER_H

#include "IParameterLoader.h"

#include "sgx_tprotected_fs.h"
#include <memory>
#include <vector>

namespace eNNclave
{
    class SgxParameterLoader : public IParameterLoader
    {
    public:
        explicit SgxParameterLoader(const std::string &parameterPath, bool write);

        ~SgxParameterLoader() noexcept override;

        void LoadParameters(float *targetBuffer, int numElements) override;
        
        void WriteParameters(float *inputs, int numElements);

    private:
        SGX_FILE *parameterFile;
        bool canWrite;
    };
} // namespace eNNclave

#endif
#ifndef SGXPARAMETERLOADER_H
#define SGXPARAMETERLOADER_H

#include "IParameterLoader.h"

#include "sgx_tprotected_fs.h"
#include <memory>

namespace eNNclave
{
    class SgxParameterLoader : public IParameterLoader
    {
    public:
        explicit SgxParameterLoader(const std::string &parameterPath);

        ~SgxParameterLoader() noexcept override;

        void LoadParameters(float *targetBuffer, int numElements) override;

    private:
        SGX_FILE *parameterFile;
    };
} // namespace eNNclave

#endif
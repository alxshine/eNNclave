#include "sgxParameterLoader.h"

#include "output.h"

using namespace eNNclave;
using namespace std;

SgxParameterLoader::SgxParameterLoader(const string &path, bool write=false): canWrite{write}{
    auto openingMode = write ? "w+" : "r";
    print_out("Opening file %s\n", path);
    parameterFile = sgx_fopen_auto_key(path.c_str(), openingMode); // TODO: handle potential file open error
}

SgxParameterLoader::~SgxParameterLoader() noexcept{
    sgx_fclose(parameterFile);
}

void SgxParameterLoader::LoadParameters(float *targetBuffer, int numElements) {
    if (canWrite)
        throw std::logic_error{"This ParameterLoader was opened for writing, cannot read"};
    
    sgx_fread(targetBuffer, sizeof(float), numElements, parameterFile);
}

unique_ptr<IParameterLoader> eNNclave::getParameterLoader(const string &parameterFile) {
    return std::unique_ptr<SgxParameterLoader>(new SgxParameterLoader(parameterFile)); // TODO: handle potential exception
}

void SgxParameterLoader::WriteParameters(float* input, int numElements){
    if (!canWrite)
        throw std::logic_error{"This ParameterLoader was opened for reading, cannot write"};

    sgx_fwrite(input, sizeof(float), numElements, parameterFile);
}
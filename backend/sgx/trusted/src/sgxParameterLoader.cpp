#include "sgxParameterLoader.h"

using namespace eNNclave;
using namespace std;

SgxParameterLoader::SgxParameterLoader(const string &path){
    parameterFile = sgx_fopen_auto_key(path.c_str(), "r");
}

SgxParameterLoader::~SgxParameterLoader() noexcept{
    sgx_fclose(parameterFile);
}

void SgxParameterLoader::LoadParameters(float *targetBuffer, int numElements) {
    sgx_fread(targetBuffer, sizeof(float), numElements, parameterFile);
}

unique_ptr<IParameterLoader> eNNclave::getParameterLoader(const string &parameterFile) {
    return std::unique_ptr<SgxParameterLoader>(new SgxParameterLoader(parameterFile));
}
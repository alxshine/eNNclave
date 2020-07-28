//
// Created by alex on 28.07.20.
//

#include "CxxParameterLoader.h"

using namespace eNNclave;
using namespace std;

void CxxParameterLoader::LoadParameters(float* targetBuffer, int numElements) {
    inputFile.read(reinterpret_cast<char*>(targetBuffer), numElements * sizeof(float));
}

CxxParameterLoader::CxxParameterLoader(const std::string &parameterPath) : inputFile(parameterPath,
                                                                                     std::ios::binary) {}

CxxParameterLoader::~CxxParameterLoader() noexcept {
    inputFile.close();
}

unique_ptr<IParameterLoader> eNNclave::getParameterLoader(const std::string &parameterFile) {
    return make_unique<CxxParameterLoader>(parameterFile);
}
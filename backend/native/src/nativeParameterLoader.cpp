//
// Created by alex on 28.07.20.
//

#include "nativeParameterLoader.h"

using namespace eNNclave;
using namespace std;

void NativeParameterLoader::LoadParameters(float* targetBuffer, int numElements) {
    inputFile.read(reinterpret_cast<char*>(targetBuffer), numElements * sizeof(float));
}

NativeParameterLoader::NativeParameterLoader(const std::string &parameterPath) : inputFile(parameterPath,
                                                                                           std::ios::binary) {}

NativeParameterLoader::~NativeParameterLoader() noexcept {
    inputFile.close();
}

unique_ptr<IParameterLoader> eNNclave::getParameterLoader(const std::string &parameterFile) {
    return make_unique<NativeParameterLoader>(parameterFile);
}
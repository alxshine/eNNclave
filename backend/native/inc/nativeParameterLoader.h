//
// Created by alex on 28.07.20.
//

#ifndef NATIVEPARAMETERLOADER_H
#define NATIVEPARAMETERLOADER_H

#include "IParameterLoader.h"

#include <string>
#include <fstream>

namespace eNNclave {
    class NativeParameterLoader : public IParameterLoader {
    public:
        explicit NativeParameterLoader(const std::string& parameterPath);

        virtual ~NativeParameterLoader() noexcept;

        void LoadParameters(float* targetBuffer, int numElements) override;

    private:
        std::ifstream inputFile;
    };
}

#endif //NATIVEPARAMETERLOADER_H

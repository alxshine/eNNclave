//
// Created by alex on 28.07.20.
//

#ifndef CXXPARAMETERLOADER_H
#define CXXPARAMETERLOADER_H

#include "IParameterLoader.h"

#include <string>
#include <fstream>

namespace eNNclave {
    class CxxParameterLoader : public IParameterLoader {
    public:
        explicit CxxParameterLoader(const std::string& parameterPath);

        virtual ~CxxParameterLoader() noexcept;

        void LoadParameters(float* targetBuffer, int numElements) override;

    private:
        std::ifstream inputFile;
    };
}

#endif //CXXPARAMETERLOADER_H

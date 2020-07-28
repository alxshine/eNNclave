//
// Created by alex on 28.07.20.
//

#ifndef IPARAMETERLOADER_H
#define IPARAMETERLOADER_H

#include <string>
#include <memory>

namespace eNNclave {
    class IParameterLoader {
    public:
        virtual void LoadParameters(float* targetBuffer, int numElements) = 0;
    };

    std::unique_ptr<IParameterLoader> getParameterLoader(const std::string &parameterFile);
}

#endif //IPARAMETERLOADER_H

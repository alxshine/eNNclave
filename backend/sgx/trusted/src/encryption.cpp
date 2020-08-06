#include "encryption.h"

#include "sgxParameterLoader.h"
#include <memory>
#include "output.h"

using namespace eNNclave;
using namespace std;

namespace
{
    std::unique_ptr<SgxParameterLoader> parameterLoader;
} // namespace

void open_encrypted_parameters()
{
    parameterLoader = std::unique_ptr<SgxParameterLoader>(new SgxParameterLoader("backend/generated/parameters.bin.aes", true)); // TODO: handle potential exception
}

int encrypt_parameters(float *target_buffer, int num_elements){
    try{
        parameterLoader->WriteParameters(target_buffer, num_elements);
    }catch(logic_error e){
        print_err(e.what());
        return 1;
    }
    return 0;
};
void close_encrypted_parameters()
{
    auto *actualLoader = parameterLoader.release();
    delete actualLoader;
}
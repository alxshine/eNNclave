#include <iostream>

#include "EnclaveHandler.h"
#include "ocalls.h"

using namespace eNNclave;
using namespace std;

int main()
{
    try
    {
        cout << "Creating EnclaveHandler" << endl;
        EnclaveHandler enclaveHandler{};
        auto path = "backend/generated/parameters.bin";
        cout << "Encrypting " << path << endl;
        enclaveHandler.encryptParameterFile(path);
    }
    catch (logic_error e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}
#include <iostream>

#include "enclave.hpp"

using namespace std;

int enclave_initialize(){
    cout << "initializing enclave" << endl;
}

int enclave_teardown(){
    cout << "destroying enclave" << endl;
}

int enclave_forward(float *m, int s, int *label){
    cout << "forward" << endl;
};

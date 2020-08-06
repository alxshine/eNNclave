#include "ocalls.h"

#include <iostream>

using namespace std;

#if defined(__cplusplus)
extern "C" {
#endif

void ocall_stdout(const char *str){
    cout << str;
}

void ocall_stderr(const char *str){
    cerr << str;
}

#if defined(__cplusplus)
}
#endif
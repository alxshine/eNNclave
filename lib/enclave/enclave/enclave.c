#include "enclave.h"
#include "enclave_t.h"
#include "matutil.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

void test(){
    print("This is the enclave :)\n");
    float a[] = {0,1,2};
    float b[] = {1,2,3};
    float ret[] = {0,0,0};
    
    matutil_multiply(a, 1, 3, b, 3, 1, ret);
    matutil_dump_matrix(ret, 1, 1);
}

int print(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_stdout_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

int print_error(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_stdout_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

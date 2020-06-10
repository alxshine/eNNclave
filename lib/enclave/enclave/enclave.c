#include "enclave.h"
#include "enclave_t.h"
#include "matutil.h"
#include "output.h"

#include "sgx_tprotected_fs.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

SGX_FILE *encrypted_parameters = NULL;

void test(){
    print_out("This is the enclave :)\n");
    float a[] = {0,1,2};
    float b[] = {1,2,3};
    float ret[] = {0,0,0};
    
    matutil_multiply(a, 1, 3, b, 3, 1, ret);
    matutil_dump_matrix(ret, 1, 1);
}

int print_out(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_stdout_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

int print_err(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_stdout_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

void open_encrypted_parameters(){
    encrypted_parameters = sgx_fopen_auto_key("encrypted_params.aes.bin", "wb+");
    if(encrypted_parameters == NULL){
        print_err("could not open encrypted parameters\n");
    }
}

int encrypt_parameters(float *buffer, int num_elements){
    int blocks_wrote = sgx_fwrite(buffer, sizeof(float), num_elements, encrypted_parameters);
    if(blocks_wrote != num_elements){
        print_err("Expected to write %d blocks, but wrote only %d\n", num_elements, blocks_wrote);
        return -1;
    }
}

void close_encrypted_parameters(){
    sgx_fclose(encrypted_parameters);
}

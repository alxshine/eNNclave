#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <pwd.h>
#include <unistd.h>
#define MAX_PATH FILENAME_MAX

#include "enclave_u.h"
#include "sgx_urts.h"
#include "wrapper.h"
#include "utils.h"

sgx_enclave_id_t enclave_id;

void ocall_print_string(const char *str) { printf("%s", str); }

void ocall_perror_string(const char *str) { fprintf(stderr, "%s", str); }

void test_enclave(void) {
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL,
                           &enclave_id, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        getchar();
        return;
    }

    test(enclave_id);

    sgx_destroy_enclave(enclave_id);

    printf("Info: Enclave tested succesfully.\n");

    printf("Enter a character before exit ...\n");
    getchar();
}

int enclave_initialize() {
    sgx_status_t ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL,
                           &enclave_id, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        getchar();
        return -1;
    }
    return 0;
}

int enclave_teardown() { return sgx_destroy_enclave(enclave_id); }

int enclave_forward(float *m, int s, int *label) {
    int sts;
    *label = -1; // so invalid results are visible
    sgx_status_t sgx_status = forward(enclave_id, &sts, m, s, label);
    if (sgx_status != SGX_SUCCESS)
    print_error_message(sgx_status);
    return sts;
}

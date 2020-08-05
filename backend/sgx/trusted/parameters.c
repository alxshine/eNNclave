#include "parameters.h"

#include "sgx_tprotected_fs.h"

SGX_FILE *param_file;

void open_parameters(){
    param_file = sgx_fopen_auto_key("encrypted_params.aes.bin", "r");
}

int load_parameters(float *target_buffer, int num_elements){
    return sgx_fread(target_buffer, sizeof(float), num_elements, param_file);
}

void close_parameters(){
    sgx_fclose(param_file);
}

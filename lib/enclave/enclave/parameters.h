#ifndef parameters_h
#define parameters_h

#include "sgx_tprotected_fs.h"

extern SGX_FILE *param_file;

void open_parameters();
int load_parameters(float *target_buffer, int num_elements);
void close_parameters();

#endif // parameters_h


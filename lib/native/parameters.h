#ifndef parameters_h
#define parameters_h

#include <stdio.h>

extern FILE *param_file;

void open_parameters();
int load_parameters(float *target_buffer, int num_elements);
void close_parameters();

#endif // parameters_h


#include "parameters.h"

#include <stdio.h>

FILE *param_file;

void open_parameters(){
    void *param_file = fopen("parameters.bin", "r");
}

int load_parameters(float *target_buffer, int num_elements){
    return fread(target_buffer, sizeof(float), num_elements, param_file);
}

void close_parameters(){
    fclose(param_file);
}

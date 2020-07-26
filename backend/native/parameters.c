#include "parameters.h"

#include <stdio.h>

FILE *param_file;

void open_parameters(){
    param_file = fopen("backend/generated/parameters.bin", "r");
    if(!param_file)
        printf("could not open parameter file");
}

int load_parameters(float *target_buffer, int num_elements){
    return fread(target_buffer, sizeof(float), num_elements, param_file);
}

void close_parameters(){
    fclose(param_file);
}

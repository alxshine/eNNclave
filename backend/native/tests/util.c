#include <stdio.h>

#include "tests.h"

int print_result(const char *name, int success)
{
    printf("TEST - %s:\t\t%s\n", name, success ? "SUCCESS" : "FAILURE");
    return success;
}

void print_separator()
{
    printf("-------------------------------------------------\n");
}
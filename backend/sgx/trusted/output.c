//
// Created by alex on 26.07.20.
//
#include "output.h"

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

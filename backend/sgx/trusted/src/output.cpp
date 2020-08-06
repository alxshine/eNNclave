//
// Created by alex on 26.07.20.
//
#include "output.h"
#include "enclave_t.h"

#include <cstdarg>
#include <cstdio>

using namespace std;

void eNNclave::print_out(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_stdout(buf);
}

void eNNclave::print_err(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_stdout(buf);
}

void eNNclave::dump_matrix(float* m, int r, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            print_out("%.09f, ", m[i * c + j]);
        }
        print_out("\n");
    }
}

void eNNclave::dump_matrix3(float* m, int h, int w, int c) {
    for (int ci = 0; ci < c; ++ci) {
        print_out("Ci=%d:\n", ci);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                print_out("%.07f, ", m[i * w * c + j * c + ci]);
            }
            print_out("\n");
        }
    }
}
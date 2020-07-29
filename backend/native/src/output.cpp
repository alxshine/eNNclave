//
// Created by alex on 28.07.20.
//

#include "output.h"

#include <iostream>
#include <cstdarg>

using namespace std;

void eNNclave::print_out(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    while (*fmt != '\0') {
        if (*fmt == 'd') {
            int i = va_arg(args, int);
            cout << i << '\n';
        } else if (*fmt == 'c') {
            // note automatic conversion to integral type
            int c = va_arg(args, int);
            cout << static_cast<char>(c) << '\n';
        } else if (*fmt == 'f') {
            double d = va_arg(args, double);
            cout << d << '\n';
        }
        ++fmt;
    }

    va_end(args);
}

void eNNclave::print_err(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    while (*fmt != '\0') {
        if (*fmt == 'd') {
            int i = va_arg(args, int);
            cerr << i << '\n';
        } else if (*fmt == 'c') {
            // note automatic conversion to integral type
            int c = va_arg(args, int);
            cerr << static_cast<char>(c) << '\n';
        } else if (*fmt == 'f') {
            double d = va_arg(args, double);
            cerr << d << '\n';
        }
        ++fmt;
    }

    va_end(args);
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
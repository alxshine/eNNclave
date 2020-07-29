#ifndef OUTPUT_H
#define OUTPUT_H

namespace eNNclave {
    void print_out(const char* fmt, ...);

    void print_err(const char* fmt, ...);

    void dump_matrix(float* m, int r, int c);

    void dump_matrix3(float* m, int h, int w, int c);

}
#endif // OUTPUT_H

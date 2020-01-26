#ifndef _enclave_h
#define _enclave_h

#if defined(__cplusplus)
extern "C" {
#endif

    void test();

    int print(const char* fmt, ...);
    int print_error(const char* fmt, ...);

    void *open_parameters();
    int load_parameters(float *target_buffer, int num_elements, void *f);
    void close_parameters(void *parameter_file);
#if defined(__cplusplus)
}
#endif

#endif

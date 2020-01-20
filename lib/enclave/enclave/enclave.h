#ifndef _enclave_h
#define _enclave_h

#if defined(__cplusplus)
extern "C" {
#endif

    void test();

    int print(const char* fmt, ...);
    int print_error(const char* fmt, ...);

#if defined(__cplusplus)
}
#endif

#endif

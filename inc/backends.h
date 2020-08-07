#ifndef BACKENDS_H
#define BACKENDS_H

#if defined(__cplusplus)
extern "C" {
#endif

typedef int FORWARD_T(float*, int, float*, int);

FORWARD_T native_forward;
FORWARD_T sgx_forward;

#if defined(__cplusplus)
}
#endif

#endif /* BACKENDS_H */

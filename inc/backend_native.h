#ifndef NATIVE_H
#define NATIVE_H

#if defined(__cplusplus)
extern "C" {
#endif

typedef int NATIVE_FORWARD_T(float*, int, float*, int);

NATIVE_FORWARD_T native_forward;

#if defined(__cplusplus)
}
#endif

#endif /* NATIVE_H */

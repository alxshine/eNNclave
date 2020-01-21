#ifndef FORWARD_H
#define FORWARD_H

#ifdef __cplusplus
extern "C" {
#endif

int forward(float *m, int s, int *label);
int print_error(const char* fmt, ...);
int print(const char* fmt, ...);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* FORWARD_H */

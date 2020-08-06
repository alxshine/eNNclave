#ifndef OCALLS_H
#define OCALLS_H

#if defined(__cplusplus)
extern "C" {
#endif

void ocall_stdout(const char *str);
void ocall_stderr(const char *str);

#if defined(__cplusplus)
}
#endif

#endif
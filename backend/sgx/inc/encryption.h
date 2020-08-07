#ifndef SGX_ENCRYPTION_H
#define SGX_ENCRYPTION_H

#if defined(__cplusplus)
extern "C" {
#endif

void open_encrypted_parameters();
int encrypt_parameters(float *target_buffer, int num_elements);
void close_encrypted_parameters();

#if defined(__cplusplus)
}
#endif

#endif
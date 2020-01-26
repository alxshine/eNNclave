#ifndef MATUTIL_H
#define MATUTIL_H

#if defined(__cplusplus)
extern "C" {
#endif

    int matutil_initialize();

    int matutil_teardown();

    void matutil_get_new_dimensions(int r1, int c1, int r2, int c2, int *rr, int *cr);

    int matutil_multiply(float *m1, int r1, int c1, float *m2, int r2, int c2, float *ret);

    int matutil_add(float *m1, int r1, int c1, float *m2, int r2, int c2, float *ret);

    int matutil_sep_conv1(float *input, int steps, int c, int f, float *depth_kernels, float *point_kernels, int ks, float *biases, float *ret);

    int matutil_conv2(float *input, int h, int w, int c, int f, float *kernels, int kh, int kw, float *biases, float *ret);

    void matutil_relu(float *m, int r, int c); // TODO: This doesn't need 2 dimensions

    void matutil_global_average_pooling_1d(float *m, int steps, int c, float *ret);

    void matutil_global_average_pooling_2d(float *m, int h, int w, int c, float *ret);

    void matutil_max_pooling_1d(float *m, int steps, int c, int pool_size, float *ret);

    void matutil_max_pooling_2d(float *m, int h, int w, int c, int pool_size, float *ret);

    int matutil_forward(float *m, int size, int *label);

    void matutil_dump_matrix(float *m, int r, int c);

    void matutil_dump_matrix3(float *m, int h, int w, int c);

    void *open_parameters();
    int load_parameters(float *target_buffer, int num_elements, void *f);
    void close_parameters(void *parameter_file);

#if defined(__cplusplus)
}
#endif

#endif /* MATUTIL_H */

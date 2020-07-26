#ifndef MATUTIL_H
#define MATUTIL_H

#define PADDING_VALID 0
#define PADDING_SAME 1

#if defined(__cplusplus)
extern "C" {
#endif

    int matutil_initialize();

    int matutil_teardown();

    void matutil_get_new_dimensions(int h1, int w1, int h2, int w2, int *rr, int *cr);

    int matutil_multiply(float *m1, int h1, int w1, float *m2, int h2, int w2, float *ret);

    int matutil_add(float *m1, int h1, int w1, float *m2, int h2, int w2, float *ret); // TODO: this doesn't need 2 dimensions

    int matutil_sep_conv1(float *input, int steps, int c, int f, float *depth_kernels, float *point_kernels, int ks, float *biases, float *ret);

    int matutil_conv2(float *input, int h, int w, int c, int f, float *kernels, int kh, int kw, float *biases, float *ret);

    int matutil_depthwise_conv2(float *input, int h, int w, int c, int padding, float *kernels, int kh, int kw, float *ret);

    void matutil_relu(float *m, int h, int w); // TODO: This doesn't need 2 dimensions

    void matutil_global_average_pooling_1d(float *m, int steps, int c, float *ret);

    void matutil_global_average_pooling_2d(float *m, int h, int w, int c, float *ret);

    void matutil_max_pooling_1d(float *m, int steps, int c, int pool_size, float *ret);

    void matutil_max_pooling_2d(float *m, int h, int w, int c, int pool_size, float *ret);

    void matutil_zero_pad2(float *m, int h, int w, int c, int top_pad, int bottom_pad, int left_pad, int right_pad, float *ret);

    int matutil_forward(float *m, int size, int *label);

    void matutil_dump_matrix(float *m, int r, int c);

    void matutil_dump_matrix3(float *m, int h, int w, int c);

#if defined(__cplusplus)
}
#endif

#endif /* MATUTIL_H */

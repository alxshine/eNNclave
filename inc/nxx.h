//
// Created by alex on 28.07.20.
//

#ifndef NXX_H
#define NXX_H

namespace eNNclave {
    void dense(const float* input, int h, int w, const float* weights, int neurons, const float* biases, float* ret);

    void sep_conv1(float* input, int steps, int c, int f, float* depth_kernels, float* point_kernels, int ks,
                   float* biases, float* ret);

    void conv2(float* input, int h, int w, int c, int f, float* kernels, int kh, int kw, float* biases, float* ret);

    void depthwise_conv2(float* input, int h, int w, int c, int padding, float* kernels, int kh, int kw, float* ret);

    void relu(float* m, int size);

    void global_average_pooling_1d(float* m, int steps, int c, float* ret);

    void global_average_pooling_2d(float* m, int h, int w, int c, float* ret);

    void max_pooling_1d(float* m, int steps, int c, int pool_size, float* ret);

    void max_pooling_2d(float* m, int h, int w, int c, int pool_size, float* ret);

    void zero_pad2(float* m, int h, int w, int c, int top_pad, int bottom_pad, int left_pad, int right_pad, float* ret);

    void dump_matrix(float* m, int r, int c);

    void dump_matrix3(float* m, int h, int w, int c);
}
#endif //NXX_H

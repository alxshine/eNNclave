//
// Created by alex on 28.07.20.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedGlobalDeclarationInspection"

#ifndef NN_H
#define NN_H

namespace eNNclave
{
    enum class Padding
    {
        SAME,
        VALID
    };

    void dense(const float *input, int h, int w, const float *weights, int neurons, const float *biases, float *ret);

    void sep_conv1(const float *input, int steps, int c, int f, const float *depth_kernels, const float *point_kernels,
                   int ks,
                   const float *biases, float *ret);

    void
    conv2(const float *input, int h, int w, int c, int f, const float *kernels, int kh, int kw, const float *biases,
          float *ret);

    void depthwise_conv2(const float *input, int h, int w, int c, Padding padding, const float *kernels, int kh, int kw,
                         float *ret);

    void relu(float *m, int size);

    void global_average_pooling_1d(const float *input, int steps, int c, float *ret);

    void global_average_pooling_2d(const float *m, int h, int w, int c, float *ret);

    void max_pooling_1d(const float *m, int steps, int c, int pool_size, float *ret);

    void max_pooling_2d(const float *m, int h, int w, int c, int pool_size, float *ret);

    void zero_pad2(const float *m, int h, int w, int c, int top_pad, int bottom_pad, int left_pad, int right_pad,
                   float *ret);

    void softmax(float *input, int size);

    void sigmoid(float *input, int size);
} // namespace eNNclave
#endif //NN_H

#pragma clang diagnostic pop
//
// Created by alex on 28.07.20.
//

#ifndef NN_H
#define NN_H

namespace eNNclave {
    enum class Padding {
        SAME,
        VALID
    };

    void dense(const float* input, int h, int w, const float* weights, int neurons, const float* biases, float* ret);

    void sep_conv1(const float* input, int steps, int c, int f, const float* depth_kernels, const float* point_kernels,
                   int ks,
                   const float* biases, float* ret);

    void
    conv2(const float* input, int h, int w, int c, int f, const float* kernels, int kh, int kw, const float* biases,
          float* ret);

    void depthwise_conv2(const float* input, int h, int w, int c, Padding padding, const float* kernels, int kh, int kw,
                         float* ret);

    void relu(const float* m, int size, float* ret);

    void global_average_pooling_1d(const float* m, int steps, int c, float* ret);

    void global_average_pooling_2d(const float* m, int h, int w, int c, float* ret);

    void max_pooling_1d(const float* m, int steps, int c, int pool_size, float* ret);

    void max_pooling_2d(const float* m, int h, int w, int c, int pool_size, float* ret);

    void zero_pad2(const float* m, int h, int w, int c, int top_pad, int bottom_pad, int left_pad, int right_pad,
                   float* ret);

    void softmax(const float* input, float* ret);

    void sigmoid(const float* input, float* ret);
}
#endif //NN_H

//
// Created by alex on 28.07.20.
//

#include "nxx.h"

void
eNNclave::dense(const float* input, int h, int w, const float* weights, int neurons, const float* biases, float* ret) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < neurons; ++j) {
            ret[i * neurons + j] = biases[j];

            for (int mul_i = 0, mul_j = 0; mul_i < w; ++mul_i, ++mul_j)
                ret[i * neurons + j] += input[i * w + mul_j] * weights[mul_i * neurons + j];
        }
    }
}
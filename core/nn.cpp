//
// Created by alex on 28.07.20.
//

#include "nn.h"
#include "output.h"

#include <iostream>

// TODO: make as much compile-time as possible

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedGlobalDeclarationInspection"

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

void
eNNclave::sep_conv1(const float* input, int steps, int c, int f, const float* depth_kernels, const float* point_kernels,
                    int ks, const float* biases, float* ret) {
    int len_ret = steps * f;
    for (int i = 0; i < len_ret; ++i)
        ret[i] = 0;

    int min_offset = ks / 2;
    if (ks % 2 == 0)
        min_offset -= 1;

    for (int i = 0; i < steps; ++i) {
        for (int di = 0; di < ks; ++di) {
            int input_i = i - min_offset + di;
            if (input_i < 0 || input_i >= steps)
                continue;

            for (int ci = 0; ci < c; ++ci)
                for (int fi = 0; fi < f; ++fi)
                    ret[i * f + fi] +=
                            input[input_i * c + ci] * depth_kernels[di * c + ci] * point_kernels[ci * f + fi];
        }

        for (int fi = 0; fi < f; ++fi)
            ret[i * f + fi] += biases[fi];
    }
}

void
eNNclave::conv2(const float* input, int h, int w, int c, int f, const float* kernels, int kh, int kw,
                const float* biases, float* ret) {
// clear ret
    int len_ret = h * w * f;
    for (int i = 0; i < len_ret; ++i) {
        ret[i] = 0;
    }

    int min_row_offset = kh / 2;
    if (kh % 2 == 0)
        min_row_offset -= 1;
    int min_col_offset = kw / 2;
    if (kw % 2 == 0)
        min_col_offset -= 1;

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int ki = 0; ki < kh; ++ki) {
                int input_i = i - min_row_offset + ki;
                if (input_i < 0 || input_i >= h)
                    continue;

                for (int kj = 0; kj < kw; ++kj) {
                    int input_j = j - min_col_offset + kj;
                    if (input_j < 0 || input_j >= w)
                        continue;

                    for (int ci = 0; ci < c; ++ci) {
                        for (int fi = 0; fi < f; ++fi) {
                            ret[i * w * f + j * f + fi] +=
                                    input[input_i * w * c + input_j * c + ci] *
                                    kernels[ki * kw * c * f + kj * c * f + ci * f + fi];
                        }
                    }
                }
            }

            for (int fi = 0; fi < f; ++fi) {
                ret[i * w * f + j * f + fi] += biases[fi];
            }
        }
    }
}

void eNNclave::relu(float* m, int size) {
    for (int i = 0; i < size; i++)
        if (m[i] < 0)
            m[i] = 0;
}

void
eNNclave::depthwise_conv2(const float* input, int h, int w, int c, Padding padding, const float* kernels, int kh,
                          int kw, float* ret) {
    int min_row_offset = kh / 2;
    if (kh % 2 == 0)
        min_row_offset -= 1;
    int min_col_offset = kw / 2;
    if (kw % 2 == 0)
        min_col_offset -= 1;

    int row_start, row_end, col_start, col_end;
    if (padding == Padding::SAME) {
        row_start = 0;
        row_end = h;
        col_start = 0;
        col_end = w;
    } else if (padding == Padding::VALID) {
        row_start = min_row_offset;
        row_end = h - min_row_offset - !(kh % 2); // fix offset in case size is even
        col_start = min_col_offset;
        col_end = h - min_col_offset - !(kw % 2);
    } else {
        print_err("Unhandled padding type\n");
        return;
    }
    int new_height = row_end - row_start;
    int new_width = col_end - col_start;

    int len_ret = new_height * new_width * c;
    for (int i = 0; i < len_ret; ++i)
        ret[i] = 0;

    for (int i = row_start; i < row_end; ++i) {
        int output_i = i - row_start;
        for (int j = col_start; j < col_end; ++j) {
            int output_j = j - col_start;
            for (int ki = 0; ki < kh; ++ki) {
                int input_i = i - min_row_offset + ki;
                if (input_i < 0 || input_i >= h)
                    continue;
                for (int kj = 0; kj < kw; ++kj) {
                    int input_j = j - min_col_offset + kj;
                    if (input_j < 0 || input_j >= w)
                        continue;
                    for (int ci = 0; ci < c; ++ci) {
                        ret[output_i * new_width * c + output_j * c + ci] +=
                                input[input_i * w * c + input_j * c + ci] *
                                kernels[ki * kw * c + kj * c + ci];
                    }
                }
            }
        }
    }
}

void eNNclave::global_average_pooling_1d(const float* input, int steps, int c, float* ret) {
    for (int ci = 0; ci < c; ++ci) {
        ret[ci] = 0;
    }

    auto fsteps = static_cast<float>(steps);

    for (int i = 0; i < steps; ++i) {
        for (int ci = 0; ci < c; ++ci) {
            ret[ci] += input[i * c + ci] / fsteps;
        }
    }
}

void eNNclave::global_average_pooling_2d(const float* m, int h, int w, int c, float* ret) {
    for (int i = 0; i < c; ++i) {
        ret[i] = 0;
    }

    auto fdiv = static_cast<float>(h * w);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int ci = 0; ci < c; ++ci) {
                ret[ci] += m[i * w * c + j * c + ci] / fdiv;
            }
        }
    }
}

void eNNclave::max_pooling_1d(const float* m, int steps, int c, int pool_size, float* ret) {
    int ret_steps = steps / pool_size;

    for (int i = 0; i < ret_steps; ++i) {
        int input_start = i * pool_size;

        for (int ci = 0; ci < c; ++ci) {
            float current_max = m[input_start * c + ci];

            for (int di = 0; di < pool_size; ++di) {
                int current_i = input_start + di;
                float to_compare = m[current_i * c + ci];
                current_max = to_compare > current_max ? to_compare : current_max;
            }

            ret[i * c + ci] = current_max;
        }
    }
}

void eNNclave::max_pooling_2d(const float* m, int h, int w, int c, int pool_size, float* ret) {
    int ret_h = h / pool_size;
    int ret_w = w / pool_size;

    for (int i = 0; i < ret_h; ++i) {
        int input_i = i * pool_size;
        for (int j = 0; j < ret_w; ++j) {
            int input_j = j * pool_size;

            for (int ci = 0; ci < c; ++ci) {
                float current_max = m[input_i * w * c + input_j * c + ci];

                for (int di = 0; di < pool_size; ++di) {
                    for (int dj = 0; dj < pool_size; ++dj) {
                        int current_i = input_i + di;
                        int current_j = input_j + dj;
                        float to_compare = m[current_i * w * c + current_j * c + ci];
                        current_max = to_compare > current_max ? to_compare : current_max;
                    }
                }

                ret[i * ret_w * c + j * c + ci] = current_max;
            }
        }
    }
}

void eNNclave::zero_pad2(const float* m, int h, int w, int c, int top_pad, int bottom_pad, int left_pad, int right_pad,
                         float* ret) {
    int new_width = w + left_pad + right_pad;
    int new_height = h + top_pad + bottom_pad;

    //top pad
    for (int i = 0; i < top_pad; ++i)
        for (int j = 0; j < new_width; ++j)
            for (int ci = 0; ci < c; ++ci)
                ret[i * new_width * c + j * c + ci] = 0;

    //copy contents
    auto content_top = top_pad;
    auto content_bottom = h + top_pad;
    auto content_left = left_pad;
    auto content_right = w + left_pad;
    for (int i = content_top, input_i = 0; i < content_bottom; ++i, ++input_i) {
        //left pad
        for (int lj = 0; lj < left_pad; ++lj)
            for (int ci = 0; ci < c; ++ci)
                ret[i * new_width * c + lj * c + ci] = 0;

        for (int j = content_left, input_j = 0; j < content_right; ++j, ++input_j)
            for (int ci = 0; ci < c; ++ci)
                ret[i * new_width * c + j * c + ci] = m[input_i * w * c + input_j * c + ci];

        //right pad
        for (int rj = content_right; rj < new_width; ++rj)
            for (int ci = 0; ci < c; ++ci)
                ret[i * new_width * c + rj * c + ci] = 0;
    }

    //bottom pad
    for (int i = content_bottom; i < new_height; ++i)
        for (int j = 0; j < new_width; ++j)
            for (int ci = 0; ci < c; ++ci)
                ret[i * new_width * c + j * c + ci] = 0;
}
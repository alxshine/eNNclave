//
// Created by alex on 26.07.20.
//

#ifndef CORE_TEST_TESTS_H
#define CORE_TEST_TESTS_H

int assert_equality(const float* a, const float* b, int n);

int assert_similarity(const float* a, const float* b, int n);

int print_result(const char* name, int success);

void print_separator();

void test_add(int* correct_cases, int* total_cases);

void test_conv2(int* correct_cases, int* total_cases);

void test_depthwise_conv2(int* correct_cases, int* total_cases);

void test_global_average_pooling1(int* correct_cases, int* total_cases);

void test_global_average_pooling2(int* correct_cases, int* total_cases);

void test_max_pool1(int* correct_cases, int* total_cases);

void test_max_pool2(int* correct_cases, int* total_cases);

void test_multiply(int* correct_cases, int* total_cases);

void test_relu(int* correct_cases, int* total_cases);

void test_sep_conv1(int* correct_cases, int* total_cases);

void test_zero_pad2(int* correct_cases, int* total_cases);

#endif //CORE_TEST_TESTS_H

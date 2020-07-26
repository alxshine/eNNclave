#include <math.h>

#include "tests.h"

// assert that two arrays of n elements each are equal
int assert_equality(const float *const a, const float *const b, int n) {
    for (int i = 0; i < n; ++i)
        if (a[i] != b[i])
            return 0;

    return 1;
}

int assert_similarity(const float *const a, const float *const b, int n) {
    for (int i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        if (fabsf(diff) > 1e-6)
            return 0;
    }

    return 1;
}
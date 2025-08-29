#ifndef FAST_MATH_CUH
#define FAST_MATH_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define LUT_SIZE 4096
__constant__ float sin_lut[LUT_SIZE];

__device__ unsigned int xorshift32(unsigned int& state) { // <-- DEVE AVERE LA &
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float random_float(unsigned int& state) { // <-- DEVE AVERE LA &
    return (float)xorshift32(state) / 4294967295.0f;
}

__device__ __forceinline__ float fast_sin(float x) {
    //if (isnan(x) || isinf(x)) return 0.0f;
    const float PI2 = 6.28318530718f;
    x = fmodf(x, PI2);
    if (x < 0.0f) x += PI2;
    const float LUT_SCALE_FACTOR = (1.0f / PI2) * (LUT_SIZE - 1);
    float pos = x * LUT_SCALE_FACTOR;
    int idx0 = static_cast<int>(pos);
    float frac = pos - idx0;
    idx0 = max(0, min(LUT_SIZE - 2, idx0));
    int idx1 = idx0 + 1;

    float s0 = sin_lut[idx0];
    float s1 = sin_lut[idx1];
    return fmaf(frac, s1 - s0, s0);
}

__device__ __forceinline__ float fast_cos(float x) {
    return fast_sin(x + 1.57079632679f);
}

void setupSinLut() {
    float temp_lut[LUT_SIZE];
    const float PI2 = 6.28318530718f;
    for (int i = 0; i < LUT_SIZE; ++i) {
        temp_lut[i] = sinf((float)i / (LUT_SIZE - 1) * PI2);
    }
    cudaMemcpyToSymbol(sin_lut, temp_lut, LUT_SIZE * sizeof(float));
}

#endif // FAST_MATH_CUH
#ifndef FAST_MATH_CUH
#define FAST_MATH_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define LUT_SIZE 4096
__constant__ float sin_lut[LUT_SIZE];

__device__ unsigned int xorshift32(unsigned int& state) {
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float random_float(unsigned int& state) {
    return (float)xorshift32(state) / 4294967295.0f;
}

__device__ __forceinline__ float fast_sin(float x) {
    return __sinf(x);
}

__device__ __forceinline__ float fast_cos(float x) {
  return __cosf(x);
}

void setupSinLut() {
    float temp_lut[LUT_SIZE];
    const float PI2 = 6.28318530718f;
    for (int i = 0; i < LUT_SIZE; ++i) {
        temp_lut[i] = sinf((float)i / (LUT_SIZE - 1) * PI2);
    }
    cudaMemcpyToSymbol(sin_lut, temp_lut, LUT_SIZE * sizeof(float));
}

#endif

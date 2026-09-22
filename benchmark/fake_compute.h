#pragma once
#include <cuda_runtime_api.h>
void launch_fake_compute(float* output, int blocks, int iterations, cudaStream_t stream);

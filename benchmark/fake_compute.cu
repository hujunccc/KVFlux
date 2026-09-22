#include "fake_compute.h"

// 使用独立 scratch 数据，传输不依赖本次计算，才有合法的 overlap 空间。
// 结果写回显存，避免编译器删除循环；这只是算术负载，不代表 transformer。
__global__ void fake_compute(float* output, int iterations) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.5f + static_cast<float>(index % 7) * 0.01f;
    for (int i = 0; i < iterations; ++i) value = fmaf(value, 1.0000001f, 0.0000001f);
    output[index] = value;
}
void launch_fake_compute(float* output, int blocks, int iterations, cudaStream_t stream) {
    fake_compute<<<blocks, 128, 0, stream>>>(output, iterations);
}

// V3.1 GPU ↔ CPU chunk 传输带宽基准。测量 pinned CPU ↔ GPU 的
// cudaMemcpyAsync 延迟和有效带宽，结果写入 CSV。
// 用法：kvflux_v3_bandwidth_benchmark [输出.csv]

#include "kvflux/v3/gpu_chunk_store.h"
#include "kvflux/v3/pinned_chunk_pool.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

struct Sample {
    std::size_t size;
    double h2d_ms, d2h_ms;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

Sample measure(std::size_t bytes, int samples) {
    kvflux::v3::GpuChunkStore gpu(1, bytes);
    kvflux::v3::PinnedChunkPool cpu(1, bytes);

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate");

    std::vector<double> h2d, d2h;
    for (int i = 0; i < samples + 2; ++i) { // 前两次作为预热丢弃
        float ms = 0;
        cuda_check(cudaEventRecord(start), "cudaEventRecord");
        cuda_check(cudaMemcpyAsync(gpu.device_ptr(0), cpu.data(0), bytes,
                                   cudaMemcpyHostToDevice), "H2D");
        cuda_check(cudaEventRecord(stop), "cudaEventRecord");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        if (i >= 2) h2d.push_back(ms);

        cuda_check(cudaEventRecord(start), "cudaEventRecord");
        cuda_check(cudaMemcpyAsync(cpu.data(0), gpu.device_ptr(0), bytes,
                                   cudaMemcpyDeviceToHost), "D2H");
        cuda_check(cudaEventRecord(stop), "cudaEventRecord");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        if (i >= 2) d2h.push_back(ms);
    }
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy");
    return Sample{bytes, median(std::move(h2d)), median(std::move(d2h))};
}

double gbps(std::size_t bytes, double ms) {
    return ms > 0 ? static_cast<double>(bytes) / (ms * 1e6) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    const std::string path = argc > 1 ? argv[1] : "gpu_cpu_bandwidth.csv";
    try {
        const std::vector<std::size_t> sizes = {256ull << 10, 1ull << 20, 4ull << 20,
                                                16ull << 20, 64ull << 20};
        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("cannot open output: " + path);
        csv << "transfer_size_bytes,h2d_latency_ms,d2h_latency_ms,"
               "h2d_bandwidth_gbps,d2h_bandwidth_gbps\n";
        for (const auto size : sizes) {
            const auto s = measure(size, 10);
            csv << s.size << ',' << s.h2d_ms << ',' << s.d2h_ms << ','
                << gbps(s.size, s.h2d_ms) << ',' << gbps(s.size, s.d2h_ms) << '\n';
            std::cout << "size=" << (size >> 20) << " MiB"
                      << " h2d=" << s.h2d_ms << " ms (" << gbps(s.size, s.h2d_ms) << " GB/s)"
                      << " d2h=" << s.d2h_ms << " ms (" << gbps(s.size, s.d2h_ms) << " GB/s)\n";
        }
        std::cout << "wrote " << path << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

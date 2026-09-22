#include "kvflux/async_transfer.h"
#include "fake_compute.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace kvflux;
using Clock = std::chrono::steady_clock;
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
double microseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
std::ofstream csv(const std::filesystem::path& file) {
    std::ofstream out(file);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::fixed << std::setprecision(6);
    return out;
}
struct Event {
    cudaEvent_t event = nullptr;
    Event() { check(cudaEventCreate(&event)); }
    ~Event() { (void)cudaEventDestroy(event); }
    void record(cudaStream_t stream) { check(cudaEventRecord(event, stream)); }
};
double elapsed(const Event& a, const Event& b) {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, a.event, b.event));
    return ms;
}

void transfer_benchmark(const std::filesystem::path& dir) {
    auto out = csv(dir / "transfers.csv");
    out << "size_bytes,direction,mode,latency_us,bandwidth_gbps,samples,iterations_per_sample\n";
    const char* names[] = {"pageable", "pinned_direct", "pinned_staged"};
    for (std::size_t size : {4096ULL, 65536ULL, 1048576ULL, 16777216ULL, 268435456ULL}) {
        GpuMemoryPool pool(1, size, 1);
        auto block = pool.allocate();
        PinnedBuffer pinned(size);
        std::vector<unsigned char> normal(size, 71), result(size);
        pinned.copy_from(normal.data(), size);
        pool.write_block(block, normal.data(), size);
        const int iterations = size <= 65536 ? 100 : (size <= 16777216 ? 5 : 1);
        for (bool write : {true, false}) {
            std::vector<double> samples[3];
            auto run = [&](int mode) {
                if (write) {
                    if (mode == 2) pinned.copy_from(normal.data(), size);
                    pool.write_block(block, mode == 0 ? normal.data() : pinned.data(), size);
                } else {
                    pool.read_block(block, mode == 0 ? result.data() : pinned.data(), size);
                    if (mode == 2) pinned.copy_to(result.data(), size);
                }
            };
            for (int m = 0; m < 3; ++m) for (int warm = 0; warm < 2; ++warm) run(m);
            // 轮换模式顺序减少温度、时钟变化对某一种模式的系统性偏置。
            for (int sample = 0; sample < 9; ++sample) {
                for (int k = 0; k < 3; ++k) {
                    const auto mode = (sample + k) % 3;
                    const auto start = Clock::now();
                    for (int i = 0; i < iterations; ++i) run(mode);
                    samples[mode].push_back(microseconds(start) / iterations);
                }
            }
            pool.read_block(block, result.data(), size);
            if (normal != result) throw std::runtime_error("transfer benchmark data mismatch");
            for (int mode = 0; mode < 3; ++mode) {
                const auto us = median(samples[mode]);
                out << size << ',' << (write ? "H2D" : "D2H") << ',' << names[mode] << ','
                    << us << ',' << size / us / 1000.0 << ",9," << iterations << '\n';
            }
        }
        pool.release(block);
        std::cout << "transfer size " << size << " bytes completed\n";
    }
}

void overlap_benchmark(const std::filesystem::path& dir, const cudaDeviceProp& prop) {
    auto out = csv(dir / "overlap.csv");
    out << "mode,size_bytes,total_ms,transfer_ms,compute_ms,effective_bandwidth_gbps,transfer_bandwidth_gbps,overlap_ms,overlap_ratio,samples\n";
    constexpr std::size_t size = 16 * 1024 * 1024;
    GpuMemoryPool pool(1, size, 1);
    AsyncTransferRuntime runtime(pool);
    auto block = pool.allocate();
    PinnedBuffer input(size);
    std::memset(input.data(), 37, size);
    // 使用另一个长寿命设备池保存 fake compute 结果，不与传输地址竞争。
    const int grid = prop.multiProcessorCount * 2;
    const std::size_t scratch_bytes = grid * 128 * sizeof(float);
    GpuMemoryPool scratch(1, scratch_bytes, 1);
    auto scratch_block = scratch.allocate();
    auto* values = static_cast<float*>(scratch.device_address(scratch_block));
    Event epoch, copy_start, copy_end, compute_start, compute_end;
    struct Sample { double total, transfer, compute, overlap, ratio; };
    std::vector<Sample> measurements[2];
    auto run = [&](bool async) {
        const auto compute = runtime.compute_stream().native_handle();
        const auto transfer = runtime.transfer_stream().native_handle();
        check(cudaDeviceSynchronize());
        const auto start = Clock::now();
        epoch.record(compute);
        check(cudaStreamWaitEvent(transfer, epoch.event, 0));
        copy_start.record(transfer);
        runtime.write_block(block, input);
        copy_end.record(transfer);
        if (!async) runtime.synchronize(); // baseline：复制完成之后才能提交计算。
        compute_start.record(compute);
        launch_fake_compute(values, grid, 500000, compute);
        check(cudaGetLastError());
        compute_end.record(compute);
        runtime.synchronize();
        runtime.compute_stream().synchronize();
        const double total = microseconds(start) / 1000.0;
        const double transfer_ms = elapsed(copy_start, copy_end);
        const double compute_ms = elapsed(compute_start, compute_end);
        // 用同一设备 event 时间轴计算两个 stream 区间的真实交集，避免猜测 overlap。
        const double begin = std::max(elapsed(epoch, copy_start), elapsed(epoch, compute_start));
        const double end = std::min(elapsed(epoch, copy_end), elapsed(epoch, compute_end));
        const double overlap = std::max(0.0, end - begin);
        return Sample{total, transfer_ms, compute_ms, overlap,
                      overlap / std::min(transfer_ms, compute_ms)};
    };
    for (int warm = 0; warm < 2; ++warm) { run(false); run(true); }
    for (int i = 0; i < 9; ++i) for (int k = 0; k < 2; ++k) {
        const auto mode = (i + k) % 2;
        measurements[mode].push_back(run(mode == 1));
    }
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<double> total, transfer, compute, overlap, ratio;
        for (const auto& s : measurements[mode]) {
            total.push_back(s.total); transfer.push_back(s.transfer); compute.push_back(s.compute);
            overlap.push_back(s.overlap); ratio.push_back(s.ratio);
        }
        const auto ms = median(total), copy_ms = median(transfer);
        out << (mode ? "async" : "sync") << ',' << size << ',' << ms << ',' << copy_ms << ','
            << median(compute) << ',' << size / ms / 1e6 << ',' << size / copy_ms / 1e6 << ','
            << median(overlap) << ',' << median(ratio) << ",9\n";
    }
    std::vector<float> result(grid * 128);
    check(cudaMemcpy(result.data(), values, scratch_bytes, cudaMemcpyDeviceToHost));
    for (float v : result) if (!std::isfinite(v) || v <= 0.5f) throw std::runtime_error("fake compute result invalid");
    std::vector<unsigned char> transferred(size);
    pool.read_block(block, transferred.data(), size);
    for (auto value : transferred) if (value != 37) throw std::runtime_error("overlap transfer mismatch");
    scratch.release(scratch_block);
    pool.release(block);
    std::cout << "sync/async overlap completed\n";
}

void batch_benchmark(const std::filesystem::path& dir) {
    auto out = csv(dir / "batch.csv");
    out << "mode,block_count,block_bytes,total_bytes,total_us,bandwidth_gbps,samples\n";
    constexpr std::size_t bytes = 1024 * 1024;
    for (std::size_t n : {1, 2, 4, 8, 16, 32}) {
        GpuMemoryPool pool(n, bytes, 1);
        AsyncTransferRuntime runtime(pool);
        PinnedBuffer input(n * bytes);
        auto* data = static_cast<unsigned char*>(input.data());
        std::vector<BlockHandle> blocks;
        for (std::size_t i = 0; i < n; ++i) {
            blocks.push_back(pool.allocate());
            std::memset(data + i * bytes, static_cast<int>(i + 1), bytes);
        }
        std::vector<double> samples[2];
        auto run = [&](bool batch) {
            const auto start = Clock::now();
            if (batch) {
                runtime.write_batch(blocks, input);
                runtime.synchronize();
            } else {
                for (std::size_t i = 0; i < n; ++i) pool.write_block(blocks[i], data + i * bytes, bytes);
            }
            return microseconds(start);
        };
        for (int warm = 0; warm < 2; ++warm) { run(false); run(true); }
        for (int i = 0; i < 9; ++i) for (int k = 0; k < 2; ++k) {
            auto mode = (i + k) % 2;
            samples[mode].push_back(run(mode == 1));
        }
        for (int mode = 0; mode < 2; ++mode) {
            const auto us = median(samples[mode]);
            out << (mode ? "async_batch" : "sync_per_block") << ',' << n << ',' << bytes << ','
                << n * bytes << ',' << us << ',' << n * bytes / us / 1000.0 << ",9\n";
        }
        PinnedBuffer output(n * bytes);
        runtime.read_batch(blocks, output);
        runtime.synchronize();
        if (std::memcmp(input.data(), output.data(), n * bytes)) throw std::runtime_error("batch benchmark mismatch");
        for (auto h : blocks) pool.release(h);
        std::cout << "batch " << n << " completed\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: kvflux_transfer_benchmark [output_directory]");
        const std::filesystem::path dir = argc == 2 ? argv[1] : "benchmark/results/local";
        std::filesystem::create_directories(dir);
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0));
        check(cudaSetDevice(0));
        check(cudaFree(nullptr)); // 将首次 CUDA context 初始化移出测量区间。
        int runtime = 0, driver = 0;
        check(cudaRuntimeGetVersion(&runtime));
        check(cudaDriverGetVersion(&driver));
        auto meta = csv(dir / "environment.txt");
        meta << "gpu=" << prop.name << "\ncompute_capability=" << prop.major << '.' << prop.minor
             << "\nsm_count=" << prop.multiProcessorCount << "\nasync_engine_count=" << prop.asyncEngineCount
             << "\ndevice_overlap=" << prop.deviceOverlap << "\nruntime=" << runtime << "\ndriver=" << driver
             << "\ncompiler=" << __VERSION__ << "\nmeasurements=median of 9 samples, 2 warmups per mode"
             << "\noverlap_transfer_bytes=16777216\nfake_compute_iterations=500000\nfake_compute_threads_per_block=128"
             << "\nfake_compute_grid=2 * sm_count\nbandwidth_unit=decimal GB/s\n";
        std::cout << "GPU: " << prop.name << '\n';
        transfer_benchmark(dir);
        overlap_benchmark(dir, prop);
        batch_benchmark(dir);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

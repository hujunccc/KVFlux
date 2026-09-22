#include "kvflux/tiered_block_manager.h"
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
constexpr std::size_t block_bytes = 1024 * 1024;
constexpr std::size_t gpu_capacity = 8, logical_capacity = 12, passes = 3;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
double us(Clock::time_point start) { return std::chrono::duration<double, std::micro>(Clock::now() - start).count(); }
double median(std::vector<double> values) { std::sort(values.begin(), values.end()); return values[values.size() / 2]; }
std::ofstream output(const std::filesystem::path& path) {
    std::ofstream file(path);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file << std::fixed << std::setprecision(6);
    return file;
}
struct Sample {
    double total_ms, stall_ms, p95_us, scheduler_ms;
    std::size_t loads, offloads, submitted, skipped;
};

Sample trial(std::size_t lookahead, int iterations, int grid) {
    TieredBlockManager manager(gpu_capacity, logical_capacity, block_bytes, 16);
    std::vector<unsigned char> host(block_bytes);
    std::vector<LogicalBlockHandle> blocks, order;
    for (std::size_t i = 0; i < logical_capacity; ++i) {
        std::fill(host.begin(), host.end(), static_cast<unsigned char>(i + 1));
        blocks.push_back(manager.create(host.data(), host.size()));
    }
    for (std::size_t pass = 0; pass < passes; ++pass) order.insert(order.end(), blocks.begin(), blocks.end());
    GpuMemoryPool scratch(1, grid * 128 * sizeof(float), 1);
    auto scratch_block = scratch.allocate();
    auto* values = static_cast<float*>(scratch.device_address(scratch_block));
    const auto stream = manager.compute_stream().native_handle();
    // 初始化、第一次 kernel JIT/launch 与 pinned 分配全部放在计时之前。
    launch_fake_compute(values, grid, std::max(iterations, 1), stream);
    manager.compute_stream().synchronize();
    const auto initial = manager.stats();
    std::vector<double> stalls;
    double stall_us = 0, scheduler_us = 0;
    const auto begin = Clock::now();
    for (std::size_t request = 0; request < order.size(); ++request) {
        const auto demand_start = Clock::now();
        auto lease = manager.acquire_gpu(order[request]);
        const auto demand_us = us(demand_start);
        stall_us += demand_us;
        stalls.push_back(demand_us);
        if (iterations) {
            launch_fake_compute(values, grid, iterations, stream);
            check(cudaGetLastError());
        }
        auto tick = Clock::now();
        if (lookahead) manager.prefetch_next(order, request, lookahead);
        scheduler_us += us(tick);
        // 与实际 scheduler 的 event polling 类似；不额外插入 sleep 帮助预取。
        // baseline 也执行相同的 query/poll 循环，计算工作量不变。
        cudaError_t ready;
        while ((ready = cudaStreamQuery(stream)) == cudaErrorNotReady) {
            tick = Clock::now();
            manager.poll();
            scheduler_us += us(tick);
        }
        check(ready);
        tick = Clock::now();
        manager.poll();
        scheduler_us += us(tick);
        // lease 在 kernel 完成后才销毁，当前块在整个计算窗口内不可迁移。
    }
    manager.wait(); // 把尾部预取收尾计入总时间，不能把成本藏到计时区间之外。
    const double total_ms = us(begin) / 1000.0;
    const auto final = manager.stats();
    std::sort(stalls.begin(), stalls.end());
    const auto p95 = stalls[static_cast<std::size_t>(std::ceil(stalls.size() * 0.95)) - 1];
    // 数据校验不计入 request stall，避免 D2H 校验本身污染访问延迟。
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        manager.read_block(blocks[i], host.data(), host.size());
        for (auto value : host) if (value != i + 1) throw std::runtime_error("prefetch benchmark data mismatch");
    }
    if (iterations) {
        std::vector<float> result(grid * 128);
        check(cudaMemcpy(result.data(), values, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (float value : result) if (!std::isfinite(value) || value <= 0.5f) throw std::runtime_error("invalid compute result");
    }
    for (auto block : blocks) manager.release(block);
    scratch.release(scratch_block);
    return {total_ms, stall_us / 1000.0, p95, scheduler_us / 1000.0,
            final.loads - initial.loads, final.offloads - initial.offloads,
            final.prefetch_submitted - initial.prefetch_submitted, final.prefetch_skipped - initial.prefetch_skipped};
}

void pressure_trace(const std::filesystem::path& dir) {
    auto file = output(dir / "pressure.csv");
    file << "created_blocks,gpu_resident,cpu_resident,transferring,gpu_free,offloads\n";
    TieredBlockManager manager(gpu_capacity, logical_capacity, block_bytes, 16);
    std::vector<unsigned char> data(block_bytes, 42);
    std::vector<LogicalBlockHandle> blocks;
    for (std::size_t i = 0; i < logical_capacity; ++i) {
        blocks.push_back(manager.create(data.data(), data.size()));
        const auto s = manager.stats();
        file << i + 1 << ',' << s.gpu_resident << ',' << s.cpu_resident << ',' << s.transferring << ','
             << s.gpu_free << ',' << s.offloads << '\n';
    }
    for (auto h : blocks) manager.release(h);
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: kvflux_prefetch_benchmark [output_directory]");
        const std::filesystem::path dir = argc == 2 ? argv[1] : "benchmark/results/prefetch-local";
        std::filesystem::create_directories(dir);
        check(cudaSetDevice(0));
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        auto metadata = output(dir / "environment.txt");
        metadata << "gpu=" << properties.name << "\ngpu_capacity=8\ncpu_backing_capacity=12\nblock_bytes=" << block_bytes
                 << "\nrequests=36 (0..11 repeated 3 times)\ninitial_state=GPU blocks 4..11, CPU blocks 0..3"
                 << "\ncompute_threads=128\ncompute_grid=" << properties.multiProcessorCount * 2
                 << "\nsamples=7, one warmup per mode/workload\npolling=host busy polling, no sleep\n";
        pressure_trace(dir);
        auto summary = output(dir / "prefetch.csv");
        auto samples = output(dir / "prefetch_samples.csv");
        summary << "compute_iterations,lookahead,requests,total_ms,request_stall_ms,mean_stall_us,p95_stall_us,scheduler_cpu_ms,samples\n";
        samples << "compute_iterations,lookahead,sample,total_ms,request_stall_ms,p95_stall_us,scheduler_cpu_ms,loads,offloads,prefetch_submitted,prefetch_skipped\n";
        const std::size_t windows[] = {0, 1, 2, 4};
        for (int iterations : {0, 500000}) {
            std::vector<Sample> measurements[4];
            for (auto n : windows) trial(n, iterations, properties.multiProcessorCount * 2);
            for (int sample = 0; sample < 7; ++sample) for (int k = 0; k < 4; ++k) {
                const auto mode = (sample + k) % 4;
                const auto s = trial(windows[mode], iterations, properties.multiProcessorCount * 2);
                measurements[mode].push_back(s);
                samples << iterations << ',' << windows[mode] << ',' << sample << ',' << s.total_ms << ',' << s.stall_ms
                        << ',' << s.p95_us << ',' << s.scheduler_ms << ',' << s.loads << ',' << s.offloads << ','
                        << s.submitted << ',' << s.skipped << '\n';
            }
            for (std::size_t mode = 0; mode < 4; ++mode) {
                std::vector<double> total, stalls, p95, scheduler;
                for (const auto& s : measurements[mode]) {
                    total.push_back(s.total_ms); stalls.push_back(s.stall_ms); p95.push_back(s.p95_us); scheduler.push_back(s.scheduler_ms);
                }
                const auto stall = median(stalls);
                summary << iterations << ',' << windows[mode] << ',' << logical_capacity * passes << ',' << median(total)
                        << ',' << stall << ',' << stall * 1000 / (logical_capacity * passes) << ',' << median(p95)
                        << ',' << median(scheduler) << ",7\n";
            }
            std::cout << "prefetch workload iterations=" << iterations << " completed\n";
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

#include "kvflux/tiered_block_manager.h"
#include "fake_compute.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace kvflux;
using Clock = std::chrono::steady_clock;
namespace {
constexpr std::size_t blocks_count = 100, block_bytes = 1024 * 1024;
constexpr int samples = 5;
struct Workload { const char* name; std::size_t gpu_capacity; };
constexpr Workload workloads[] = {{"A", 100}, {"B", 80}, {"C", 30}};
constexpr std::size_t windows[] = {0, 1, 4};
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::ofstream output(const std::filesystem::path& path) {
    std::ofstream file(path);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file << std::fixed << std::setprecision(6);
    return file;
}
std::vector<std::size_t> access_order(const std::string& trace) {
    std::vector<std::size_t> ids;
    for (int phase = 0; phase < 3; ++phase) {
        if (trace == "locality") {
            // 同一轨迹：60 块热集合重复 3 次，然后访问 40 块冷集合。
            for (int repeat = 0; repeat < 3; ++repeat)
                for (std::size_t id = 0; id < 60; ++id) ids.push_back(id);
            for (std::size_t id = 60; id < blocks_count; ++id) ids.push_back(id);
        } else {
            for (std::size_t id = 0; id < blocks_count; ++id) ids.push_back(id);
        }
    }
    return ids;
}
TransferDirectionMetrics delta(TransferDirectionMetrics end, const TransferDirectionMetrics& begin) {
    end.bytes -= begin.bytes;
    end.batches -= begin.batches;
    end.device_ms -= begin.device_ms;
    return end;
}
CacheMetrics delta(CacheMetrics end, const CacheMetrics& begin) {
    end.transfers.gpu_to_cpu = delta(end.transfers.gpu_to_cpu, begin.transfers.gpu_to_cpu);
    end.transfers.cpu_to_gpu = delta(end.transfers.cpu_to_gpu, begin.transfers.cpu_to_gpu);
    end.transfers.failed_batches -= begin.transfers.failed_batches;
    end.offload_count -= begin.offload_count;
    end.load_count -= begin.load_count;
    end.prefetch_count -= begin.prefetch_count;
    end.prefetch_skipped -= begin.prefetch_skipped;
    end.prefetch_hits -= begin.prefetch_hits;
    end.prefetch_misses -= begin.prefetch_misses;
    end.prefetch_late -= begin.prefetch_late;
    end.prefetch_unused -= begin.prefetch_unused;
    end.demand_count -= begin.demand_count;
    end.gpu_demand_hits -= begin.gpu_demand_hits;
    end.request_stall_ms -= begin.request_stall_ms;
    return end; // gauges 保留结束时快照，不相减。
}
struct Sample {
    CacheMetrics metrics;
    double total_ms, p95_stall_us, scheduler_ms;
    std::uint64_t initial_offloads;
};
Sample trial(const Workload& workload, const std::vector<std::size_t>& ids,
             std::size_t lookahead, int compute_iterations, int grid) {
    TieredBlockManager manager(workload.gpu_capacity, blocks_count, block_bytes, 16);
    std::vector<unsigned char> host(block_bytes);
    std::vector<LogicalBlockHandle> blocks, order;
    for (std::size_t i = 0; i < blocks_count; ++i) {
        std::fill(host.begin(), host.end(), static_cast<unsigned char>(i + 1));
        blocks.push_back(manager.create(host.data(), host.size()));
    }
    for (auto id : ids) order.push_back(blocks[id]);
    GpuMemoryPool scratch(1, grid * 128 * sizeof(float), 1);
    const auto scratch_block = scratch.allocate();
    auto* values = static_cast<float*>(scratch.device_address(scratch_block));
    const auto stream = manager.compute_stream().native_handle();
    launch_fake_compute(values, grid, std::max(compute_iterations, 1), stream);
    check(cudaGetLastError());
    manager.compute_stream().synchronize();
    const auto initial = manager.metrics();
    std::vector<double> stalls;
    stalls.reserve(ids.size());
    double scheduler_ms = 0;
    const auto begin = Clock::now();
    for (std::size_t i = 0; i < order.size(); ++i) {
        const auto start = Clock::now();
        auto lease = manager.acquire_gpu(order[i]);
        stalls.push_back(ms(start) * 1000);
        if (compute_iterations) {
            launch_fake_compute(values, grid, compute_iterations, stream);
            check(cudaGetLastError());
        }
        auto tick = Clock::now();
        if (lookahead) manager.prefetch_next(order, i, lookahead);
        scheduler_ms += ms(tick);
        cudaError_t ready;
        while ((ready = cudaStreamQuery(stream)) == cudaErrorNotReady) {
            tick = Clock::now();
            manager.poll();
            scheduler_ms += ms(tick);
        }
        check(ready);
        tick = Clock::now();
        manager.poll();
        scheduler_ms += ms(tick);
    }
    manager.wait(); // 包含最后一批未完成迁移。
    const auto total_ms = ms(begin);
    const auto measured = delta(manager.metrics(), initial);
    require(measured.demand_count == order.size(), "demand metric mismatch");
    require(measured.gpu_blocks_used + measured.gpu_blocks_free == workload.gpu_capacity, "GPU capacity mismatch");
    require(measured.transferring_blocks == 0 && measured.transfers.failed_batches == 0, "unfinished/failed transfer");
    require(measured.transfers.gpu_to_cpu.bytes == measured.offload_count * block_bytes, "D2H accounting mismatch");
    require(measured.transfers.cpu_to_gpu.bytes == measured.load_count * block_bytes, "H2D accounting mismatch");
    require(measured.prefetch_count == measured.prefetch_hits + measured.prefetch_late +
            measured.prefetch_unused + measured.prefetch_outstanding, "prefetch outcome mismatch");
    require(measured.gpu_demand_hits + measured.prefetch_misses == measured.demand_count, "demand outcome mismatch");
    if (workload.gpu_capacity == blocks_count)
        require(measured.offload_count == 0 && measured.load_count == 0, "workload A unexpectedly migrated");
    // 最终回读不计入 measured metrics，也不计入时间。
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        manager.read_block(blocks[i], host.data(), host.size());
        for (auto byte : host) require(byte == i + 1, "KV data mismatch");
    }
    std::vector<float> result(grid * 128);
    check(cudaMemcpy(result.data(), values, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (auto value : result) require(std::isfinite(value) && value > 0.5f, "compute data mismatch");
    for (auto block : blocks) manager.release(block);
    scratch.release(scratch_block);
    std::sort(stalls.begin(), stalls.end());
    return {measured, total_ms, stalls[static_cast<std::size_t>(std::ceil(stalls.size() * 0.95)) - 1],
            scheduler_ms, initial.offload_count};
}
void row(std::ostream& out, const Workload& w, const std::string& trace, int iterations,
         std::size_t window, int sample, const Sample& s) {
    const auto& m = s.metrics;
    const auto& d = m.transfers.gpu_to_cpu;
    const auto& h = m.transfers.cpu_to_gpu;
    out << w.name << ',' << trace << ',' << iterations << ',' << window << ',' << sample << ','
        << m.demand_count << ',' << block_bytes << ',' << m.gpu_blocks_total << ',' << m.gpu_blocks_used << ','
        << m.gpu_blocks_free << ',' << m.cpu_cached_blocks << ',' << m.transferring_blocks << ','
        << s.initial_offloads << ',' << s.total_ms << ',' << m.request_stall_ms << ',' << s.p95_stall_us << ','
        << s.scheduler_ms << ',' << m.offload_count << ',' << m.load_count << ',' << m.prefetch_count << ','
        << m.prefetch_skipped << ',' << m.prefetch_hits << ',' << m.prefetch_misses << ',' << m.prefetch_late << ','
        << m.prefetch_unused << ',' << m.prefetch_outstanding << ',' << m.gpu_demand_hits << ','
        << d.bytes << ',' << h.bytes << ',' << d.device_ms << ',' << h.device_ms << ','
        << d.latency_us() << ',' << h.latency_us() << ',' << d.bandwidth_gbps() << ',' << h.bandwidth_gbps() << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: kvflux_cache_benchmark [output_directory]");
        const std::filesystem::path dir = argc == 2 ? argv[1] : "benchmark/results/v1-local";
        std::filesystem::create_directories(dir);
        check(cudaSetDevice(0));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0));
        int runtime = 0, driver = 0;
        check(cudaRuntimeGetVersion(&runtime));
        check(cudaDriverGetVersion(&driver));
        auto metadata = output(dir / "environment.txt");
        const auto now = std::time(nullptr);
        metadata << "utc=" << std::put_time(std::gmtime(&now), "%FT%TZ")
                 << "\ngpu=" << prop.name << "\nruntime=" << runtime << "\ndriver=" << driver
                 << "\ncompiler=" << __VERSION__ << "\nversion=1.10.0\nlogical_blocks=100\nblock_bytes=" << block_bytes
                 << "\ngpu_capacities=A:100,B:80,C:30\nsamples=" << samples
                 << "\nlookahead=0,1,4\ncompute_iterations=0,500000\ncompute_grid=" << prop.multiProcessorCount * 2
                 << "\ncompute_threads=128\ncompute_data=independent scratch (not attention)"
                 << "\nwarmup=one full trial per configuration\ninitial_state=GPU last capacity blocks, CPU others"
                 << "\nlocality=([0..59]*3 + [60..99])*3; 660 demands"
                 << "\nscan=[0..99]*3; 300 demands\npolling=host busy polling, no sleep"
                 << "\nsetup_and_final_validation=excluded from timing and counter deltas"
                 << "\nrequest_stall=runtime acquire_gpu host wall time"
                 << "\ntransfer_latency=CUDA event time per completed batch (one block here)"
                 << "\ntransfer_bandwidth=completed bytes / summed event time; decimal GB/s\n";
        auto csv = output(dir / "cache_samples.csv");
        csv << "workload,trace,compute_iterations,lookahead,sample,requests,block_bytes,gpu_blocks_total,gpu_blocks_used,"
               "gpu_blocks_free,cpu_cached_blocks,transferring_blocks,initial_offloads,total_ms,request_stall_ms,"
               "p95_stall_us,scheduler_cpu_ms,offload_count,load_count,prefetch_count,prefetch_skipped,prefetch_hits,"
               "prefetch_misses,prefetch_late,prefetch_unused,prefetch_outstanding,gpu_demand_hits,gpu_to_cpu_bytes,"
               "cpu_to_gpu_bytes,gpu_to_cpu_ms,cpu_to_gpu_ms,gpu_to_cpu_latency_us,cpu_to_gpu_latency_us,"
               "gpu_to_cpu_bandwidth_gbps,cpu_to_gpu_bandwidth_gbps\n";
        for (const std::string trace : {"locality", "scan"}) {
            const auto ids = access_order(trace);
            auto order_csv = output(dir / (trace + "_order.csv"));
            order_csv << "request,block_id\n";
            for (std::size_t i = 0; i < ids.size(); ++i) order_csv << i << ',' << ids[i] << '\n';
            for (int iterations : {0, 500000}) {
                for (const auto& w : workloads) for (auto window : windows)
                    trial(w, ids, window, iterations, prop.multiProcessorCount * 2);
                for (int sample = 0; sample < samples; ++sample) {
                    // 同时轮换容量和预取模式的测试顺序。
                    for (int k = 0; k < 9; ++k) {
                        const auto mode = (sample + k) % 9;
                        const auto& w = workloads[mode / 3];
                        const auto window = windows[mode % 3];
                        const auto s = trial(w, ids, window, iterations, prop.multiProcessorCount * 2);
                        row(csv, w, trace, iterations, window, sample, s);
                    }
                    csv.flush();
                    std::cout << trace << " compute=" << iterations << " sample=" << sample + 1 << '/' << samples
                              << " completed; all data/metric checks passed" << std::endl;
                }
            }
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

// V3.3 迁移与计算重叠基准。证明 MigrationManager 的 transfer stream 与
// compute stream 上的 kernel 真正并发执行，而不只是调用了异步 API：
// 先分别测"纯计算"和"纯迁移"耗时，再测串行（计算完再迁移）与重叠
//（计算在 compute stream、迁移经 MigrationManager 同时进行）的总耗时。
// 若重叠真实发生，overlapped ≈ max(compute, migration) 而
// sequential ≈ compute + migration。
// 用法：kvflux_v3_overlap_benchmark [输出.csv]

#include "kvflux/v3/cache_directory.h"
#include "kvflux/v3/chunk_file_store.h"
#include "kvflux/v3/gpu_chunk_store.h"
#include "kvflux/v3/migration_manager.h"
#include "kvflux/v3/pinned_chunk_pool.h"
#include "kvflux/v3/tier.h"
#include "kvflux/physical_block_pool.h"

#include "fake_compute.h"

#include <cuda_runtime_api.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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

double wall_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since)
        .count();
}

struct Rig {
    static constexpr std::size_t CHUNKS = 8;

    kvflux::PhysicalBlockPool pool;
    kvflux::v3::GpuTier gpu;
    kvflux::v3::CpuTier cpu;
    kvflux::v3::NvmeTier nvme;
    kvflux::v3::CacheDirectory dir;
    kvflux::v3::GpuChunkStore gpu_store;
    kvflux::v3::PinnedChunkPool cpu_pool;
    kvflux::v3::ChunkFileStore store;
    kvflux::v3::MigrationManager mgr;

    Rig(std::size_t chunk_bytes, const std::string& root)
        : pool(CHUNKS, 16), gpu(pool, chunk_bytes), cpu(CHUNKS, chunk_bytes),
          nvme(1, 1), dir(gpu, cpu, nvme), gpu_store(CHUNKS, chunk_bytes),
          cpu_pool(CHUNKS, chunk_bytes),
          store(root + "/overlap_" + std::to_string(chunk_bytes), 1ull << 30),
          mgr(dir, gpu_store, cpu_pool, store, 1) {}

    kvflux::v3::CacheKey key(std::size_t i) const {
        return kvflux::v3::make_cache_key("model-bench", 1, kvflux::DType::Float16,
                                          0, 32, kvflux::Tokens{7, static_cast<int>(i)});
    }

    // 所有 chunk 只在 GPU 上有 READY 副本。
    void seed_on_gpu() {
        std::vector<unsigned char> fill(gpu_store.chunk_bytes(), 0xAB);
        for (std::size_t i = 0; i < CHUNKS; ++i) {
            dir.create(key(i));
            const auto loc = dir.place(key(i), kvflux::v3::StorageTier::GPU);
            cuda_check(cudaMemcpy(gpu_store.device_ptr(loc), fill.data(), fill.size(),
                                  cudaMemcpyHostToDevice), "seed H2D");
        }
    }

    // 把全部 chunk GPU→CPU 迁移一遍，返回墙钟毫秒；结束后 CPU 副本被移除，
    // 恢复"仅 GPU READY"的初始状态。
    double run_migrations() {
        std::vector<kvflux::v3::TaskID> ids;
        for (std::size_t i = 0; i < CHUNKS; ++i) {
            ids.push_back(mgr.submit(key(i), kvflux::v3::StorageTier::CPU));
        }
        const auto begin = std::chrono::steady_clock::now();
        for (const auto id : ids) {
            if (mgr.wait(id) != kvflux::v3::MigrationState::COMPLETED) {
                throw std::runtime_error("migration failed: " + mgr.info(id).error);
            }
        }
        const double ms = wall_ms(begin);
        for (std::size_t i = 0; i < CHUNKS; ++i) {
            dir.remove(key(i), kvflux::v3::StorageTier::CPU);
        }
        return ms;
    }
};

struct Row {
    std::size_t chunk_bytes;
    std::size_t chunks;
    int kernels;
    double compute_ms, migration_ms, sequential_ms, overlapped_ms;
};

Row measure(std::size_t chunk_bytes, const std::string& root, float* compute_buffer,
            cudaStream_t compute_stream) {
    Rig rig(chunk_bytes, root);
    rig.seed_on_gpu();

    constexpr int BLOCKS = 256;
    constexpr int ITERATIONS = 1 << 20;

    // 纯迁移基线，同时用于校准计算量。
    const double migration_ms = rig.run_migrations();

    // 单个 kernel 的耗时，用来把计算总量校准到与迁移相当。
    cuda_check(cudaStreamSynchronize(compute_stream), "sync");
    auto begin = std::chrono::steady_clock::now();
    launch_fake_compute(compute_buffer, BLOCKS, ITERATIONS, compute_stream);
    cuda_check(cudaStreamSynchronize(compute_stream), "sync");
    const double kernel_ms = wall_ms(begin);
    const int kernels =
        std::max(1, static_cast<int>(std::lround(migration_ms / kernel_ms)));

    // 纯计算基线。
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kernels; ++i) {
        launch_fake_compute(compute_buffer, BLOCKS, ITERATIONS, compute_stream);
    }
    cuda_check(cudaStreamSynchronize(compute_stream), "sync");
    const double compute_ms = wall_ms(begin);

    // 串行：先算完，再迁移。
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kernels; ++i) {
        launch_fake_compute(compute_buffer, BLOCKS, ITERATIONS, compute_stream);
    }
    cuda_check(cudaStreamSynchronize(compute_stream), "sync");
    const double sequential_ms = wall_ms(begin) + rig.run_migrations();

    // 重叠：kernel 挂在 compute stream 上，迁移走 MigrationManager 的
    // transfer stream，两者同时推进。
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kernels; ++i) {
        launch_fake_compute(compute_buffer, BLOCKS, ITERATIONS, compute_stream);
    }
    std::vector<kvflux::v3::TaskID> ids;
    for (std::size_t i = 0; i < Rig::CHUNKS; ++i) {
        ids.push_back(rig.mgr.submit(rig.key(i), kvflux::v3::StorageTier::CPU));
    }
    for (const auto id : ids) {
        if (rig.mgr.wait(id) != kvflux::v3::MigrationState::COMPLETED) {
            throw std::runtime_error("migration failed: " + rig.mgr.info(id).error);
        }
    }
    cuda_check(cudaStreamSynchronize(compute_stream), "sync");
    const double overlapped_ms = wall_ms(begin);
    for (std::size_t i = 0; i < Rig::CHUNKS; ++i) {
        rig.dir.remove(rig.key(i), kvflux::v3::StorageTier::CPU);
    }

    return Row{chunk_bytes, Rig::CHUNKS, kernels,
               compute_ms, migration_ms, sequential_ms, overlapped_ms};
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    const std::string path = argc > 1 ? argv[1] : "async_overlap.csv";
    const std::string root =
        "/tmp/kvflux_v3_overlap_bench_" + std::to_string(::getpid());
    try {
        cudaStream_t compute_stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
                   "cudaStreamCreate");
        float* compute_buffer = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&compute_buffer),
                              256 * 128 * sizeof(float)), "cudaMalloc");

        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("cannot open output: " + path);
        csv << "chunk_size_bytes,num_chunks,num_kernels,compute_ms,migration_ms,"
               "sequential_ms,overlapped_ms,speedup\n";
        for (const std::size_t size : {4ull << 20, 16ull << 20, 64ull << 20}) {
            const auto row = measure(size, root, compute_buffer, compute_stream);
            const double speedup = row.sequential_ms / row.overlapped_ms;
            csv << row.chunk_bytes << ',' << row.chunks << ',' << row.kernels << ','
                << row.compute_ms << ',' << row.migration_ms << ','
                << row.sequential_ms << ',' << row.overlapped_ms << ',' << speedup
                << '\n';
            std::cout << "chunk=" << (row.chunk_bytes >> 20) << " MiB x" << row.chunks
                      << " compute=" << row.compute_ms << " ms"
                      << " migration=" << row.migration_ms << " ms"
                      << " sequential=" << row.sequential_ms << " ms"
                      << " overlapped=" << row.overlapped_ms << " ms"
                      << " speedup=" << speedup << "x\n";
        }
        cudaFree(compute_buffer);
        cudaStreamDestroy(compute_stream);
        std::cout << "wrote " << path << '\n';
        std::filesystem::remove_all(root);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}

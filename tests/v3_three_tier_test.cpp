// V3.2 三级迁移闭环：GPU → CPU → NVMe → CPU → GPU 数据完全一致（7.5、7.6-3）。

#include "kvflux/v3/chunk_file_store.h"
#include "kvflux/v3/gpu_chunk_store.h"
#include "kvflux/v3/gpu_cpu_transfer.h"
#include "kvflux/v3/nvme_transfer.h"
#include "kvflux/v3/pinned_chunk_pool.h"

#include <cuda_runtime_api.h>

#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

using kvflux::Tokens;
using kvflux::v3::CacheDirectory;
using kvflux::v3::CacheKey;
using kvflux::v3::ChunkFileStore;
using kvflux::v3::ChunkState;
using kvflux::v3::CpuTier;
using kvflux::v3::GpuTier;
using kvflux::v3::NvmeTier;
using kvflux::v3::StorageTier;

namespace {

constexpr std::size_t SLOTS = 8;
constexpr std::size_t BYTES = 1 << 20;

const std::string& test_root() {
    static const std::string root =
        "/tmp/kvflux_v3_three_tier_" + std::to_string(::getpid());
    return root;
}

struct Fixture {
    kvflux::PhysicalBlockPool gpu_pool{SLOTS, 16};
    GpuTier gpu{gpu_pool, BYTES};
    CpuTier cpu{SLOTS, BYTES};
    NvmeTier nvme{SLOTS, BYTES};
    CacheDirectory dir{gpu, cpu, nvme};
    kvflux::v3::GpuChunkStore gpu_store{SLOTS, BYTES};
    kvflux::v3::PinnedChunkPool cpu_pool{SLOTS, BYTES};
    ChunkFileStore store{test_root(), 64ull << 20};
    kvflux::v3::GpuCpuTransfer gpu_cpu{dir, gpu_store, cpu_pool};
    kvflux::v3::NvmeTransfer cpu_nvme{dir, store, cpu_pool};
};

std::vector<unsigned char> random_bytes(std::size_t n) {
    std::mt19937_64 rng(0x5EED);
    std::vector<unsigned char> out(n);
    for (auto& b : out) b = static_cast<unsigned char>(rng());
    return out;
}

std::vector<unsigned char> read_gpu(const kvflux::v3::GpuChunkStore& store,
                                    kvflux::v3::TierLocation loc) {
    std::vector<unsigned char> out(store.chunk_bytes());
    const auto result = cudaMemcpy(out.data(), store.device_ptr(loc), out.size(),
                                   cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    return out;
}

} // namespace

// 7.5 的完整闭环：GPU Block → CPU Buffer → NVMe Chunk → CPU Buffer → 新 GPU Block。
void three_tier_round_trip() {
    Fixture f;
    const auto key = kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                                0, 32, Tokens{42, 43, 44});
    const auto data = random_bytes(BYTES);

    // 起点：KV 在 GPU 上（经 CPU pinned 写入）。
    f.dir.create(key);
    auto cpu_loc = f.dir.place(key, StorageTier::CPU);
    std::memcpy(f.cpu_pool.data(cpu_loc.slot), data.data(), BYTES);
    f.gpu_cpu.begin_onload(key);
    f.gpu_cpu.finish_onload(key, true);
    const auto first_gpu = f.dir.at(key).gpu->location;
    CHECK(f.dir.replica_count(key) == 1);

    // 显存不足：GPU → CPU。
    f.gpu_cpu.begin_offload(key);
    f.gpu_cpu.finish_offload(key, true);
    CHECK(f.dir.replica_count(key) == 1);
    CHECK(f.dir.at(key).cpu->state == ChunkState::READY);
    CHECK(f.gpu.allocated_chunks() == 0);

    // CPU 内存不足：CPU → NVMe。文件里是真实 KV 字节，不是 token 或请求元数据。
    f.cpu_nvme.offload(key, 32, true);
    CHECK(f.dir.replica_count(key) == 1);
    CHECK(f.dir.at(key).nvme->state == ChunkState::READY);
    {
        const auto expected = kvflux::v3::make_file_meta(key, 0, BYTES);
        std::vector<unsigned char> from_file(BYTES);
        kvflux::v3::ChunkFileMeta meta;
        CHECK(f.store.read(f.dir.at(key).nvme->location.slot, expected,
                           from_file.data(), BYTES, &meta));
        CHECK(from_file == data && meta.valid_token_count == 32);
    }

    // 其他请求占用了 A 原来的物理页（7.5 中 17 → 91 的原因）。
    const auto other = kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                                  0, 32, Tokens{7, 8, 9});
    auto other_data = random_bytes(BYTES);
    for (auto& b : other_data) b ^= 0x5A;
    f.dir.create(other);
    const auto other_cpu = f.dir.place(other, StorageTier::CPU);
    std::memcpy(f.cpu_pool.data(other_cpu.slot), other_data.data(), BYTES);
    f.gpu_cpu.begin_onload(other);
    f.gpu_cpu.finish_onload(other, true);
    CHECK(f.dir.at(other).gpu->location.slot == first_gpu.slot);

    // 请求再次需要这份 KV：NVMe → CPU → GPU。
    f.cpu_nvme.onload(key, true);
    CHECK(f.dir.at(key).cpu->state == ChunkState::READY);
    f.gpu_cpu.begin_onload(key);
    f.gpu_cpu.finish_onload(key, true);

    const auto& record = f.dir.at(key);
    CHECK(f.dir.replica_count(key) == 1);
    CHECK(record.gpu->location.slot != first_gpu.slot); // 物理页完全不同
    CHECK(read_gpu(f.gpu_store, record.gpu->location) == data); // 字节完全一致
    CHECK(read_gpu(f.gpu_store, f.dir.at(other).gpu->location) == other_data);
    // A 的 CPU/NVMe 副本已释放；B 独占自己的 GPU 页。
    CHECK(f.cpu.allocated_chunks() == 0 && f.nvme.allocated_chunks() == 0);
    CHECK(f.gpu.allocated_chunks() == 2);
}

int main() {
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    std::filesystem::remove_all(test_root());
    try {
        three_tier_round_trip();
        std::cout << "V3.2 three-tier round trip: GPU->CPU->NVMe->CPU->GPU byte-exact\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(test_root());
        return 1;
    }
    std::filesystem::remove_all(test_root());
}

#include "kvflux/v3/gpu_chunk_store.h"
#include "kvflux/v3/gpu_cpu_transfer.h"
#include "kvflux/v3/pinned_chunk_pool.h"

#include <cuda_runtime_api.h>

#include <cstring>
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
using kvflux::v3::ChunkState;
using kvflux::v3::CpuTier;
using kvflux::v3::GpuTier;
using kvflux::v3::NvmeTier;
using kvflux::v3::StorageTier;
using kvflux::v3::TierLocation;

namespace {

constexpr std::size_t SLOTS = 8;
constexpr std::size_t BYTES = 1 << 20; // 1 MiB chunk

struct Fixture {
    kvflux::PhysicalBlockPool gpu_pool{SLOTS, 16};
    GpuTier gpu{gpu_pool, BYTES};
    CpuTier cpu{SLOTS, BYTES};
    NvmeTier nvme{SLOTS, BYTES};
    CacheDirectory dir{gpu, cpu, nvme};
    kvflux::v3::GpuChunkStore gpu_store{SLOTS, BYTES};
    kvflux::v3::PinnedChunkPool cpu_pool{SLOTS, BYTES};
    kvflux::v3::GpuCpuTransfer xfer{dir, gpu_store, cpu_pool};
};

CacheKey make_key(std::uint64_t tag) {
    return kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                      0, 32, Tokens{1, 2, 3, static_cast<kvflux::Token>(tag)});
}

// xorshift 固定模式：确定性、逐字节可复现。
std::vector<unsigned char> pattern(std::uint64_t seed, std::size_t bytes) {
    std::vector<unsigned char> out(bytes);
    std::uint64_t x = seed | 1;
    for (auto& b : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        b = static_cast<unsigned char>(x);
    }
    return out;
}

std::vector<unsigned char> read_gpu(const kvflux::v3::GpuChunkStore& store, TierLocation loc) {
    std::vector<unsigned char> out(store.chunk_bytes());
    const auto result = cudaMemcpy(out.data(), store.device_ptr(loc), out.size(),
                                   cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    return out;
}

// CPU 放 pattern → onload → 释放 CPU 副本；chunk 只剩 GPU READY 副本。
void seed_on_gpu(Fixture& f, const CacheKey& key, const std::vector<unsigned char>& data) {
    f.dir.create(key);
    const auto loc = f.dir.place(key, StorageTier::CPU);
    std::memcpy(f.cpu_pool.data(loc.slot), data.data(), BYTES);
    f.xfer.begin_onload(key);
    f.xfer.finish_onload(key, true);
    CHECK(!f.dir.at(key).cpu.has_value());
    CHECK(f.dir.at(key).gpu->state == ChunkState::READY);
}

void gpu_to_cpu_test() {
    Fixture f;
    const auto key = make_key(1);
    const auto data = pattern(0xA17, BYTES);
    seed_on_gpu(f, key, data);

    f.xfer.begin_offload(key);
    f.xfer.finish_offload(key, false);
    const auto& record = f.dir.at(key);
    CHECK(record.cpu->state == ChunkState::READY);
    CHECK(record.gpu->state == ChunkState::READY);
    CHECK(std::memcmp(f.cpu_pool.data(record.cpu->location.slot), data.data(), BYTES) == 0);
}

void cpu_to_gpu_test() {
    Fixture f;
    const auto key = make_key(2);
    const auto data = pattern(0xB22, BYTES);
    f.dir.create(key);
    const auto cpu_loc = f.dir.place(key, StorageTier::CPU);
    std::memcpy(f.cpu_pool.data(cpu_loc.slot), data.data(), BYTES);

    f.xfer.begin_onload(key);
    f.xfer.finish_onload(key, false);
    const auto gpu_loc = f.dir.at(key).gpu->location;
    CHECK(read_gpu(f.gpu_store, gpu_loc) == data);
}

void round_trip_test() {
    Fixture f;
    const auto key = make_key(3);
    const auto data = pattern(0xC33, BYTES);
    seed_on_gpu(f, key, data);
    const auto before = f.dir.at(key).gpu->location;

    f.xfer.begin_offload(key);
    f.xfer.finish_offload(key, true);
    CHECK(!f.dir.at(key).gpu.has_value());
    CHECK(f.gpu.allocated_chunks() == 0); // GPU Block 已归还

    f.xfer.begin_onload(key);
    f.xfer.finish_onload(key, true);
    const auto& record = f.dir.at(key);
    CHECK(!record.cpu.has_value() && record.gpu->state == ChunkState::READY);
    // LIFO 复用同一物理页，generation 已递增。
    CHECK(record.gpu->location.slot == before.slot);
    CHECK(record.gpu->location.generation != before.generation);
    CHECK(read_gpu(f.gpu_store, record.gpu->location) == data);
}

// GPU block 位置改变后数据不变：逻辑身份不变，attention 输入字节不变。
void remapping_test() {
    Fixture f;
    const auto a = make_key(4), b = make_key(5);
    const auto data_a = pattern(0xD44, BYTES), data_b = pattern(0xE55, BYTES);
    seed_on_gpu(f, a, data_a);
    const auto old_gpu = f.dir.at(a).gpu->location;

    f.xfer.begin_offload(a);
    f.xfer.finish_offload(a, true); // GPU Block 释放，CPU 副本保留

    seed_on_gpu(f, b, data_b); // B 占用 A 原来的物理页
    CHECK(f.dir.at(b).gpu->location.slot == old_gpu.slot);

    f.xfer.begin_onload(a); // A 恢复到新的物理页
    f.xfer.finish_onload(a, true);
    const auto& record = f.dir.at(a);
    CHECK(record.gpu->location.slot != old_gpu.slot);
    CHECK(read_gpu(f.gpu_store, record.gpu->location) == data_a);
    CHECK(read_gpu(f.gpu_store, f.dir.at(b).gpu->location) == data_b);
}

void block_reuse_test() {
    Fixture f;
    const auto a = make_key(6), b = make_key(7);
    seed_on_gpu(f, a, pattern(0xF66, BYTES));
    const auto stale = f.dir.at(a).gpu->location;

    f.xfer.begin_offload(a);
    f.xfer.finish_offload(a, true);

    seed_on_gpu(f, b, pattern(0x077, BYTES));
    CHECK(f.dir.at(b).gpu->location.slot == stale.slot); // 原 block 被安全复用
    // 旧位置因 generation 失配被拒绝，无法误释放 B 的页。
    bool caught = false;
    try { f.gpu.release(stale); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
    CHECK(read_gpu(f.gpu_store, f.dir.at(b).gpu->location) == pattern(0x077, BYTES));
}

void pending_transfer_test() {
    Fixture f;
    const auto key = make_key(8);
    seed_on_gpu(f, key, pattern(0x188, BYTES));

    f.xfer.begin_offload(key);
    CHECK(f.xfer.has_pending(key));
    const auto& in_flight = f.dir.at(key);
    CHECK(in_flight.gpu->state == ChunkState::COPYING);
    CHECK(in_flight.cpu->state == ChunkState::COPYING);
    CHECK(in_flight.pin_count == 1);

    // 传输未完成：GPU 副本不能移除，chunk 不能销毁，不能重复发起。
    bool logic = false;
    try { f.dir.remove(key, StorageTier::GPU); } catch (const std::logic_error&) { logic = true; }
    CHECK(logic);
    logic = false;
    try { f.dir.destroy(key); } catch (const std::logic_error&) { logic = true; }
    CHECK(logic);
    logic = false;
    try { f.xfer.begin_offload(key); } catch (const std::logic_error&) { logic = true; }
    CHECK(logic);
    logic = false;
    try { f.xfer.finish_onload(key, false); } catch (const std::logic_error&) { logic = true; }
    CHECK(logic); // 方向不匹配

    f.xfer.finish_offload(key, true);
    CHECK(!f.xfer.has_pending(key));
    const auto& done = f.dir.at(key);
    CHECK(!done.gpu.has_value());
    CHECK(done.cpu->state == ChunkState::READY && done.pin_count == 0);
}

void multi_block_test() {
    Fixture f;
    const std::size_t n = 4;
    std::vector<CacheKey> keys;
    for (std::size_t i = 0; i < n; ++i) {
        keys.push_back(make_key(100 + i));
        seed_on_gpu(f, keys.back(), pattern(0x900 + i, BYTES));
    }
    for (const auto& key : keys) f.xfer.begin_offload(key); // 允许多个并发在途
    for (std::size_t i = 0; i < n; ++i) f.xfer.finish_offload(keys[i], true);
    CHECK(f.gpu.allocated_chunks() == 0 && f.cpu.allocated_chunks() == n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& record = f.dir.at(keys[i]);
        const auto expected = pattern(0x900 + i, BYTES);
        CHECK(std::memcmp(f.cpu_pool.data(record.cpu->location.slot),
                          expected.data(), BYTES) == 0);
    }
}

// FP16/BF16 随机原始字节 GPU → CPU → GPU 逐位一致。
void random_round_trip_test() {
    Fixture f;
    std::mt19937_64 rng(0xC0FFEE);
    std::vector<unsigned char> data(BYTES);
    for (auto& b : data) b = static_cast<unsigned char>(rng());

    const auto key = make_key(9);
    seed_on_gpu(f, key, data);
    f.xfer.begin_offload(key);
    f.xfer.finish_offload(key, true);
    f.xfer.begin_onload(key);
    f.xfer.finish_onload(key, true);
    CHECK(read_gpu(f.gpu_store, f.dir.at(key).gpu->location) == data);
}

} // namespace

int main() {
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    try {
        gpu_to_cpu_test();
        cpu_to_gpu_test();
        round_trip_test();
        remapping_test();
        block_reuse_test();
        pending_transfer_test();
        multi_block_test();
        random_round_trip_test();
        std::cout << "V3.1 GPU<->CPU migration: all transfers byte-exact\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

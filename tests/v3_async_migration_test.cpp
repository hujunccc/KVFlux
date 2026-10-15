// V3.3 验收：100+ 个异步迁移任务与真实 paged attention 并发执行，
// 全部结果正确、无重复释放、无 use-after-free、BlockTable 映射不变。

#include "kvflux/v3/chunk_file_store.h"
#include "kvflux/v3/gpu_chunk_store.h"
#include "kvflux/v3/migration_manager.h"
#include "kvflux/v3/nvme_transfer.h"
#include "kvflux/v3/pinned_chunk_pool.h"
#include "kvflux/v2/paged_attention.h"
#include "kvflux/v2/paged_kv_storage.h"
#include "kvflux/v2/sequence_state.h"

#include <cuda_runtime_api.h>

#include <unistd.h>

#include <cstring>
#include <deque>
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
using kvflux::v3::MigrationState;
using kvflux::v3::NvmeTier;
using kvflux::v3::StorageTier;
using kvflux::v3::TaskID;
using kvflux::v3::TierLocation;

namespace {

const std::string& test_root() {
    static const std::string root =
        "/tmp/kvflux_v3_async_test_" + std::to_string(::getpid());
    return root;
}

std::vector<unsigned char> pattern(std::uint64_t seed, std::size_t bytes) {
    std::vector<unsigned char> out(bytes);
    std::uint64_t x = seed | 1;
    for (auto& b : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        b = static_cast<unsigned char>(x);
    }
    return out;
}

std::vector<unsigned char> read_gpu(const kvflux::v3::GpuChunkStore& store,
                                    TierLocation loc) {
    std::vector<unsigned char> out(store.chunk_bytes());
    const auto result = cudaMemcpy(out.data(), store.device_ptr(loc), out.size(),
                                   cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    return out;
}

// ---- 100 任务 × 并发 attention 的主验收 ----

constexpr std::size_t POOL_BLOCKS = 64;
constexpr std::size_t CPU_SLOTS = 16;
constexpr std::size_t BYTES = 1 << 20;
constexpr int NUM_CHUNKS = 100;

struct MainFixture {
    kvflux::PhysicalBlockPool pool{POOL_BLOCKS, 16};
    GpuTier gpu{pool, BYTES};
    CpuTier cpu{CPU_SLOTS, BYTES};
    NvmeTier nvme{128, BYTES};
    CacheDirectory dir{gpu, cpu, nvme};
    kvflux::v3::GpuChunkStore gpu_store{POOL_BLOCKS, BYTES};
    kvflux::v3::PinnedChunkPool cpu_pool{CPU_SLOTS, BYTES};
    ChunkFileStore store{test_root() + "/main", 512ull << 20};
    kvflux::v3::NvmeTransfer sync_nvme{dir, store, cpu_pool};
    kvflux::v3::MigrationManager mgr{dir, gpu_store, cpu_pool, store, 4};
};

CacheKey chunk_key(int i) {
    return kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                      0, 32, Tokens{1000, i});
}

// v2 侧真实 paged attention：请求持有 3 个物理页，迁移并发期间不得受影响。
struct AttentionSide {
    kvflux::v2::PagedKVStorage storage; // 与 v3 共用同一个 PhysicalBlockPool
    kvflux::v2::SequenceState request;
    float* query = nullptr;
    float* output = nullptr;
    std::vector<float> reference;
    std::vector<kvflux::PhysicalBlockID> block_ids;

    AttentionSide(kvflux::PhysicalBlockPool& pool)
        : storage(pool, 2, 64, kvflux::DType::Float32), request(7, pool) {}
};

void fill_attention_kv(AttentionSide& a) {
    a.request.append_tokens(48);
    const auto& layout = a.storage.layout();
    const auto page_bytes = layout.page_bytes();
    for (std::size_t i = 0; i < 3; ++i) {
        const auto handle = a.request.block_table().handle(i);
        std::vector<unsigned char> k(page_bytes), v(page_bytes);
        for (std::size_t j = 0; j < page_bytes; ++j) {
            k[j] = static_cast<unsigned char>((i * 131 + j) % 251);
            v[j] = static_cast<unsigned char>((i * 57 + j) % 241);
        }
        cudaMemcpy(a.storage.page_address(kvflux::KVKind::Key, handle), k.data(),
                   page_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(a.storage.page_address(kvflux::KVKind::Value, handle), v.data(),
                   page_bytes, cudaMemcpyHostToDevice);
    }
    for (std::size_t i = 0; i < 3; ++i) {
        a.block_ids.push_back(a.request.block_table().physical_id(i));
    }
}

void run_attention(AttentionSide& a) {
    kvflux::v2::paged_attention(a.storage, a.request, a.query, a.output, 1, 4);
}

void hundred_tasks_with_concurrent_attention() {
    MainFixture f;
    AttentionSide attn(f.pool);
    fill_attention_kv(attn);

    const std::size_t floats = 4 * 64;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&attn.query), floats * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(reinterpret_cast<void**>(&attn.output), floats * sizeof(float)) == cudaSuccess);
    std::vector<float> q(floats);
    for (std::size_t i = 0; i < floats; ++i) q[i] = static_cast<float>((i % 13) - 6) * 0.25f;
    cudaMemcpy(attn.query, q.data(), floats * sizeof(float), cudaMemcpyHostToDevice);
    run_attention(attn);
    attn.reference.resize(floats);
    cudaMemcpy(attn.reference.data(), attn.output, floats * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 预置 100 个 chunk 到 NVMe（同步 setup，不在验收范围内）。
    std::vector<std::vector<unsigned char>> truth(NUM_CHUNKS);
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        truth[i] = pattern(0xC0 + i, BYTES);
        const auto key = chunk_key(i);
        f.dir.create(key);
        const auto loc = f.dir.place(key, StorageTier::CPU);
        std::memcpy(f.cpu_pool.data(loc.slot), truth[i].data(), BYTES);
        f.sync_nvme.offload(key, 32, true);
    }
    CHECK(f.cpu.allocated_chunks() == 0 && f.nvme.allocated_chunks() == NUM_CHUNKS);
    const auto pool_free_before = f.pool.available();

    struct InFlight { TaskID id; int chunk; bool to_gpu; };
    std::deque<int> remaining;
    for (int i = 0; i < NUM_CHUNKS; ++i) remaining.push_back(i);
    std::vector<InFlight> inflight;
    int completed = 0, attention_calls = 0, iterations = 0, tasks_submitted = 0;

    while (completed < NUM_CHUNKS) {
        while (!remaining.empty()) { // 资源允许就继续提交；满则稍后重试
            try {
                const int c = remaining.front();
                const auto id = f.mgr.submit(chunk_key(c), StorageTier::CPU);
                inflight.push_back({id, c, false});
                remaining.pop_front();
                ++tasks_submitted;
            } catch (const kvflux::CapacityError&) { break; }
        }
        for (auto it = inflight.begin(); it != inflight.end();) {
            const auto state = f.mgr.poll(it->id);
            if (state == MigrationState::COMPLETED) {
                const auto key = chunk_key(it->chunk);
                if (!it->to_gpu) {
                    it->id = f.mgr.submit(key, StorageTier::GPU);
                    it->to_gpu = true;
                    ++tasks_submitted;
                    ++it;
                } else {
                    // 迁移完成后 GPU 字节必须与预置数据逐位一致。
                    const auto gpu_loc = f.dir.at(key).gpu->location;
                    CHECK(read_gpu(f.gpu_store, gpu_loc) == truth[it->chunk]);
                    f.dir.remove(key, StorageTier::GPU);
                    f.dir.remove(key, StorageTier::CPU);
                    ++completed;
                    it = inflight.erase(it);
                }
            } else {
                CHECK(state == MigrationState::RUNNING || state == MigrationState::QUEUED);
                ++it;
            }
        }
        if (++iterations % 3 == 0) { // 迁移 churn 期间周期性执行 attention
            run_attention(attn);
            ++attention_calls;
        }
    }
    CHECK(tasks_submitted == 2 * NUM_CHUNKS);
    CHECK(attention_calls > 0);

    // attention 输出与迁移前逐位一致，BlockTable 映射不变。
    run_attention(attn);
    std::vector<float> after(floats);
    cudaMemcpy(after.data(), attn.output, floats * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(after == attn.reference);
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(attn.request.block_table().physical_id(i) == attn.block_ids[i]);
    }
    // 无泄漏、无重复释放：槽位全部归还，池水位复原。
    CHECK(f.gpu.allocated_chunks() == 0 && f.cpu.allocated_chunks() == 0);
    CHECK(f.pool.available() == pool_free_before);
    CHECK(f.mgr.pending_count() == 0);
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        const auto& record = f.dir.at(chunk_key(i));
        CHECK(!record.gpu && !record.cpu && record.nvme.has_value());
        CHECK(record.pin_count == 0);
    }
    cudaFree(attn.query);
    cudaFree(attn.output);
}

// ---- 小夹具：取消 / 失败 / 有界性 / 状态可见性 ----

struct SmallFixture {
    kvflux::PhysicalBlockPool pool{8, 16};
    GpuTier gpu{pool, 8 << 20};
    CpuTier cpu{4, 8 << 20};
    NvmeTier nvme{8, 8 << 20};
    CacheDirectory dir{gpu, cpu, nvme};
    kvflux::v3::GpuChunkStore gpu_store{8, 8 << 20};
    kvflux::v3::PinnedChunkPool cpu_pool{4, 8 << 20};
    ChunkFileStore store{test_root() + "/small", 256ull << 20};
    kvflux::v3::NvmeTransfer sync_nvme{dir, store, cpu_pool};
    kvflux::v3::MigrationManager mgr{dir, gpu_store, cpu_pool, store, 1}; // 单 I/O 线程
};

CacheKey small_key(int i) {
    return kvflux::v3::make_cache_key("model-y", 1, kvflux::DType::BFloat16,
                                      0, 32, Tokens{2000, i});
}

void seed_cpu(SmallFixture& f, const CacheKey& key, std::uint64_t seed) {
    f.dir.create(key);
    const auto loc = f.dir.place(key, StorageTier::CPU);
    const auto data = pattern(seed, 8 << 20);
    std::memcpy(f.cpu_pool.data(loc.slot), data.data(), data.size());
}

void cancelled_queued_task_frees_resources() {
    SmallFixture f;
    seed_cpu(f, small_key(1), 0x1);
    seed_cpu(f, small_key(2), 0x2);

    const auto a = f.mgr.submit(small_key(1), StorageTier::NVME, 32);
    const auto b = f.mgr.submit(small_key(2), StorageTier::NVME, 32);
    // 单 I/O 线程正忙于 A（8 MiB 写盘），B 还在队列里，可以立即取消。
    CHECK(f.mgr.cancel(b));
    CHECK(f.mgr.info(b).state == MigrationState::CANCELLED);
    const auto& record_b = f.dir.at(small_key(2));
    CHECK(record_b.cpu->state == ChunkState::READY); // 源恢复
    CHECK(!record_b.nvme.has_value() && record_b.pin_count == 0);
    CHECK(f.cpu.allocated_chunks() == 2 && f.nvme.allocated_chunks() == 1); // 仅剩 A 的目标槽

    CHECK(f.mgr.wait(a) == MigrationState::COMPLETED);
    CHECK(f.nvme.allocated_chunks() == 1);
}

void cancelled_running_cuda_task_keeps_buffers_until_safe() {
    SmallFixture f;
    seed_cpu(f, small_key(3), 0x3);
    auto id = f.mgr.submit(small_key(3), StorageTier::GPU);
    CHECK(f.mgr.wait(id) == MigrationState::COMPLETED);
    f.dir.remove(small_key(3), StorageTier::CPU); // chunk 只在 GPU

    id = f.mgr.submit(small_key(3), StorageTier::CPU); // GPU → CPU，8 MiB
    CHECK(f.mgr.cancel(id)); // RUNNING：记录取消，DMA 仍须安全完成
    const auto state = f.mgr.wait(id);
    CHECK(state == MigrationState::CANCELLED);
    const auto& record = f.dir.at(small_key(3));
    CHECK(!record.cpu.has_value()); // 目标副本被丢弃而非标 READY
    CHECK(record.gpu->state == ChunkState::READY && record.pin_count == 0);
    // 源数据未被取消影响。
    CHECK(read_gpu(f.gpu_store, record.gpu->location) == pattern(0x3, 8 << 20));
}

void failed_task_releases_resources() {
    SmallFixture f;
    seed_cpu(f, small_key(4), 0x4);
    f.sync_nvme.offload(small_key(4), 32, true);
    const auto slot = f.dir.at(small_key(4)).nvme->location.slot;
    CHECK(::unlink(f.store.path_for(slot).c_str()) == 0); // 外部删除文件

    const auto before_cpu = f.cpu.allocated_chunks();
    const auto id = f.mgr.submit(small_key(4), StorageTier::CPU); // NVMe → CPU 必失败
    CHECK(f.mgr.wait(id) == MigrationState::FAILED);
    CHECK(!f.mgr.info(id).error.empty());
    const auto& record = f.dir.at(small_key(4));
    CHECK(!record.cpu.has_value());                    // 目标槽位已回收
    CHECK(record.nvme->state == ChunkState::READY);    // 源恢复
    CHECK(record.pin_count == 0);
    CHECK(f.cpu.allocated_chunks() == before_cpu);
}

void bounded_pinned_memory_backpressure() {
    SmallFixture f; // CPU 只有 4 个槽位
    for (int i = 0; i < 3; ++i) seed_cpu(f, small_key(10 + i), 0x10 + i);
    seed_cpu(f, small_key(20), 0x20);
    f.sync_nvme.offload(small_key(20), 32, true); // 让出 CPU 槽位，数据留在 NVMe
    seed_cpu(f, small_key(21), 0x21);
    CHECK(f.cpu.allocated_chunks() == 4);
    // 第 5 个 NVMe→CPU 任务在提交时即被拒绝，不会排队吃 pinned 内存。
    bool caught = false;
    try { f.mgr.submit(small_key(20), StorageTier::CPU); }
    catch (const kvflux::CapacityError&) { caught = true; }
    CHECK(caught);
    CHECK(f.mgr.pending_count() == 0);
    CHECK(f.dir.at(small_key(20)).nvme->state == ChunkState::READY);
}

void not_ready_until_copy_completes() {
    SmallFixture f;
    seed_cpu(f, small_key(30), 0x30);
    const auto up = f.mgr.submit(small_key(30), StorageTier::GPU);
    CHECK(f.mgr.wait(up) == MigrationState::COMPLETED);
    f.dir.remove(small_key(30), StorageTier::CPU);

    const auto id = f.mgr.submit(small_key(30), StorageTier::CPU); // 8 MiB D2H
    // 结构保证：finalize 只发生在 poll/wait 确认 event 之后。
    // 提交后立刻看，目标副本一定还是 COPYING，不可能提前 READY。
    CHECK(f.dir.at(small_key(30)).cpu->state == ChunkState::COPYING);
    CHECK(f.mgr.poll(id) != MigrationState::COMPLETED ||
          f.dir.at(small_key(30)).cpu->state == ChunkState::READY);
    CHECK(f.mgr.wait(id) == MigrationState::COMPLETED);
    CHECK(f.dir.at(small_key(30)).cpu->state == ChunkState::READY);
}

} // namespace

int main() {
    int count = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    std::filesystem::remove_all(test_root());
    try {
        hundred_tasks_with_concurrent_attention();
        cancelled_queued_task_frees_resources();
        cancelled_running_cuda_task_keeps_buffers_until_safe();
        failed_task_releases_resources();
        bounded_pinned_memory_backpressure();
        not_ready_until_copy_completes();
        std::cout << "V3.3 async migration: 200 tasks with concurrent attention, "
                     "cancel/failure/backpressure all safe\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(test_root());
        return 1;
    }
    std::filesystem::remove_all(test_root());
}

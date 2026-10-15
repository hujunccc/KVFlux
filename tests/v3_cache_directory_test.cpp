#include "kvflux/v3/cache_directory.h"

#include <iostream>
#include <stdexcept>
#include <string>

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

namespace {

struct Tiers {
    kvflux::PhysicalBlockPool gpu_pool{32, 16};
    GpuTier gpu{gpu_pool, 1 << 20};
    CpuTier cpu{16, 1 << 20};
    NvmeTier nvme{32, 1 << 20};
    CacheDirectory directory{gpu, cpu, nvme};
};

CacheKey key_a() {
    return kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                      0, 32, Tokens{10, 20, 30, 40});
}

} // namespace

// 5.6 主流程：Chunk A 依次 GPU Block 17 → CPU Buffer → NVMe File Location。
void migration_updates_directory() {
    Tiers t;
    const auto a = key_a();
    t.directory.create(a);
    CHECK(t.directory.contains(a) && t.directory.size() == 1);
    CHECK(t.directory.replica_count(a) == 0);
    CHECK(t.directory.at(a).version == 1);

    // 先占掉 GPU 前 17 页，让 Chunk A 恰好落在 Block 17。
    GpuTier& gpu = t.gpu;
    kvflux::v3::TierLocation taken[17];
    for (auto& loc : taken) loc = gpu.allocate();
    const auto gpu_loc = t.directory.place(a, StorageTier::GPU);
    CHECK(gpu_loc.slot == 17);
    {
        const auto& r = t.directory.at(a);
        CHECK(r.gpu.has_value() && r.gpu->location == gpu_loc);
        CHECK(r.gpu->state == ChunkState::READY);
        CHECK(!r.cpu && !r.nvme);
        CHECK(t.directory.replica_count(a) == 1);
    }

    // GPU → CPU：先移除 GPU 副本（槽位经目录还给池），再登记 CPU 副本。
    // CPU 池前 8 槽被占，Chunk A 落在 Buffer 8。
    t.directory.remove(a, StorageTier::GPU);
    for (auto& loc : taken) gpu.release(loc);
    CpuTier& cpu = t.cpu;
    kvflux::v3::TierLocation cpu_taken[8];
    for (auto& loc : cpu_taken) loc = cpu.allocate();
    const auto cpu_loc = t.directory.place(a, StorageTier::CPU, ChunkState::LOADING);
    CHECK(cpu_loc.slot == 8);
    {
        const auto& r = t.directory.at(a);
        CHECK(!r.gpu && r.cpu.has_value() && !r.nvme);
        CHECK(r.cpu->location == cpu_loc && r.cpu->state == ChunkState::LOADING);
        CHECK(t.directory.replica_count(a) == 1);
    }
    t.directory.set_state(a, StorageTier::CPU, ChunkState::READY);
    CHECK(t.directory.at(a).cpu->state == ChunkState::READY);

    // CPU → NVMe：NVMe 前 24 槽被占，Chunk A 落在 File Location 24。
    t.directory.remove(a, StorageTier::CPU);
    for (auto& loc : cpu_taken) cpu.release(loc);
    NvmeTier& nvme = t.nvme;
    kvflux::v3::TierLocation nvme_taken[24];
    for (auto& loc : nvme_taken) loc = nvme.allocate();
    const auto nvme_loc = t.directory.place(a, StorageTier::NVME);
    CHECK(nvme_loc.slot == 24);
    {
        const auto& r = t.directory.at(a);
        CHECK(!r.gpu && !r.cpu && r.nvme.has_value());
        CHECK(r.nvme->location == nvme_loc && r.nvme->state == ChunkState::READY);
        CHECK(t.directory.replica_count(a) == 1);
        CHECK(r.version == 1); // 纯位置迁移不改版本
    }
    t.directory.remove(a, StorageTier::NVME);
    for (auto& loc : nvme_taken) nvme.release(loc);
    t.directory.destroy(a);
    CHECK(t.directory.size() == 0);
}

// GPU BlockID 被重用不影响 CacheKey：同一物理编号先后属于两个不同 chunk。
void physical_id_reuse_does_not_alias_keys() {
    Tiers t;
    const auto a = key_a();
    const auto b = kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                              0, 32, Tokens{10, 20, 30, 41});
    t.directory.create(a);
    t.directory.create(b);

    const auto loc_a = t.directory.place(a, StorageTier::GPU);
    t.directory.remove(a, StorageTier::GPU);
    // 池的 free list 立刻复用同一物理编号。
    const auto loc_b = t.directory.place(b, StorageTier::GPU);
    CHECK(loc_b.slot == loc_a.slot && loc_b.generation != loc_a.generation);

    // 两个身份互不干扰；A 没有任何副本，B 独占该物理页。
    CHECK(t.directory.replica_count(a) == 0);
    const auto& rec_b = t.directory.at(b);
    CHECK(rec_b.gpu.has_value() && rec_b.gpu->location == loc_b);
    CHECK(t.directory.replica_count(b) == 1);

    t.directory.remove(b, StorageTier::GPU);
    t.directory.destroy(a);
    t.directory.destroy(b);
}

// 不产生重复或悬空引用：同层重复登记、双层副本、逐层独立状态、销毁保护。
void duplicate_and_dangling_references_are_rejected() {
    Tiers t;
    const auto a = key_a();
    t.directory.create(a);

    const auto gpu_loc = t.directory.place(a, StorageTier::GPU);
    bool logic = false;
    try { t.directory.place(a, StorageTier::GPU); }
    catch (const std::logic_error&) { logic = true; }
    CHECK(logic); // 同层第二份副本被拒绝

    t.directory.place(a, StorageTier::CPU, ChunkState::COPYING);
    t.directory.place(a, StorageTier::NVME, ChunkState::LOADING);
    CHECK(t.directory.replica_count(a) == 3);

    // 各层副本状态独立：GPU READY、CPU COPYING、NVMe LOADING 可同时成立。
    const auto& r = t.directory.at(a);
    CHECK(r.gpu->state == ChunkState::READY);
    CHECK(r.cpu->state == ChunkState::COPYING);
    CHECK(r.nvme->state == ChunkState::LOADING);

    // 引用/钉住计数与访问时间。
    t.directory.add_ref(a);
    t.directory.add_ref(a);
    t.directory.pin(a);
    const auto before = t.directory.at(a).last_access;
    t.directory.touch(a);
    CHECK(t.directory.at(a).last_access >= before);
    t.directory.bump_version(a);
    CHECK(t.directory.at(a).version == 2);

    // 有副本/引用/钉住时拒绝销毁，防止悬空引用。
    logic = false;
    try { t.directory.destroy(a); }
    catch (const std::logic_error&) { logic = true; }
    CHECK(logic);
    t.directory.remove(a, StorageTier::GPU);
    t.directory.remove(a, StorageTier::CPU);
    t.directory.remove(a, StorageTier::NVME);
    logic = false;
    try { t.directory.destroy(a); }
    catch (const std::logic_error&) { logic = true; }
    CHECK(logic); // 仍有 ref/pin
    t.directory.release_ref(a);
    t.directory.release_ref(a);
    t.directory.unpin(a);
    t.directory.destroy(a);
    CHECK(t.directory.size() == 0);

    // 移除时槽位已还给对应层：GPU 池恢复全空。
    CHECK(t.gpu.allocated_chunks() == 0 && t.gpu_pool.available() == t.gpu_pool.capacity());
    CHECK(t.cpu.allocated_chunks() == 0 && t.nvme.allocated_chunks() == 0);
    (void)gpu_loc;
}

// 每层容量与副本数量统计：目录视图与层自身计数一致。
void per_tier_stats_match_directory_view() {
    Tiers t;
    const auto a = key_a();
    const auto b = kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                              0, 32, Tokens{7, 7, 7});
    const auto c = kvflux::v3::make_cache_key("model-y", 2, kvflux::DType::Float32,
                                              0, 1, Tokens{7, 7, 7});
    t.directory.create(a);
    t.directory.create(b);
    t.directory.create(c);

    t.directory.place(a, StorageTier::GPU);
    t.directory.place(b, StorageTier::GPU);
    t.directory.place(a, StorageTier::CPU);
    t.directory.place(c, StorageTier::NVME);

    const auto gs = t.directory.stats(StorageTier::GPU);
    CHECK(gs.capacity == 32 && gs.replicas == 2 && gs.allocated_chunks == 2);
    CHECK(gs.used_bytes == 2 * (1 << 20) && gs.available == 30);
    CHECK(gs.available_bytes == 30 * (1 << 20));

    const auto cs = t.directory.stats(StorageTier::CPU);
    CHECK(cs.capacity == 16 && cs.replicas == 1 && cs.allocated_chunks == 1);
    CHECK(cs.used_bytes == (1 << 20) && cs.available == 15);

    const auto ns = t.directory.stats(StorageTier::NVME);
    CHECK(ns.capacity == 32 && ns.replicas == 1 && ns.allocated_chunks == 1);

    // 不同 model/布局版本/dtype/层范围的相同前缀是不同的身份。
    CHECK(!(b == c));
    t.directory.remove(a, StorageTier::GPU);
    t.directory.remove(b, StorageTier::GPU);
    t.directory.remove(a, StorageTier::CPU);
    t.directory.remove(c, StorageTier::NVME);
    t.directory.destroy(a);
    t.directory.destroy(b);
    t.directory.destroy(c);
}

// 哈希碰撞不误命中：注入常数哈希后，不同前缀仍是不同身份。
void hash_collision_does_not_merge_keys() {
    Tiers t;
    const auto p = kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                              0, 32, Tokens{1, 2, 3});
    auto q = p;
    q.prefix = Tokens{1, 2, 4};
    q.prefix_hash = p.prefix_hash; // 强制碰撞
    CHECK(!(p == q));
    t.directory.create(p);
    t.directory.create(q);
    CHECK(t.directory.size() == 2);
    t.directory.place(p, StorageTier::GPU);
    t.directory.place(q, StorageTier::GPU);
    CHECK(t.directory.at(p).gpu->location.slot != t.directory.at(q).gpu->location.slot);
}

int main() {
    try {
        migration_updates_directory();
        physical_id_reuse_does_not_alias_keys();
        duplicate_and_dangling_references_are_rejected();
        per_tier_stats_match_directory_view();
        hash_collision_does_not_merge_keys();
        std::cout << "V3 cache directory: 3-tier locations, states, identity, stats\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

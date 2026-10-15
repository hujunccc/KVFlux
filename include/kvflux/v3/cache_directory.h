#pragma once

#include "kvflux/block_manager.h" // Tokens、prefix_hash
#include "kvflux/kv_layout.h"     // DType
#include "kvflux/v3/tier.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace kvflux::v3 {

enum class ChunkState { READY, COPYING, LOADING, EVICTING, ERROR };

// 一个 KVChunk 的逻辑身份，与任何物理位置无关：GPU BlockID 被回收重用
// 不会改变 CacheKey。除 token 前缀外还包含模型、布局版本、dtype 和层范围，
// 避免不同配置的缓存被错误复用。prefix_hash 复用 v0 的 FNV-1a；prefix 原文
// 保留用于精确比较，哈希碰撞不会误命中。
struct CacheKey {
    std::string model_id;
    std::uint32_t layout_version = 0;
    DType dtype = DType::Float16;
    // chunk 覆盖的层范围 [first_layer, last_layer)；单层版本取 [l, l+1)。
    std::uint32_t first_layer = 0;
    std::uint32_t last_layer = 0;
    std::uint64_t prefix_hash = 0;
    Tokens prefix;

    bool operator==(const CacheKey& other) const noexcept {
        return model_id == other.model_id && layout_version == other.layout_version &&
               dtype == other.dtype && first_layer == other.first_layer &&
               last_layer == other.last_layer && prefix_hash == other.prefix_hash &&
               prefix == other.prefix;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const noexcept;
};

// 由 token 前缀构造 CacheKey，prefix_hash 自动计算，避免哈希与原文不一致。
CacheKey make_cache_key(std::string model_id, std::uint32_t layout_version, DType dtype,
                        std::uint32_t first_layer, std::uint32_t last_layer,
                        Tokens prefix);

// 一个 tier 上的副本：位置加该副本自己的状态。chunk 身份与各层副本状态
// 分开管理，同一 chunk 可以同时 GPU: READY、CPU: READY、NVMe: COPYING。
struct Replica {
    TierLocation location;
    ChunkState state = ChunkState::READY;
};

struct ChunkRecord {
    CacheKey key;
    std::uint64_t version = 1;
    std::size_t ref_count = 0;
    std::size_t pin_count = 0;
    std::chrono::steady_clock::time_point last_access;
    std::optional<Replica> gpu;
    std::optional<Replica> cpu;
    std::optional<Replica> nvme;
};

// V3 的元数据核心：登记每份 KV 的逻辑身份、三级位置、各副本状态、引用与
// 钉住计数和最近访问时间。副本槽位只通过目录向对应 Tier 申请/归还，因此
// 不会产生重复或悬空引用。不传输任何数据；单线程，调用方负责同步。
class CacheDirectory {
public:
    // 三个参数必须分别是 GPU/CPU/NVME kind 的 Tier，且都比目录活得久。
    CacheDirectory(Tier& gpu, Tier& cpu, Tier& nvme);

    // 登记 chunk 身份，version 从 1 开始，无任何副本。key 已存在时抛
    // invalid_argument，保证不产生重复身份。
    void create(const CacheKey& key);
    bool contains(const CacheKey& key) const;
    // 未登记时抛 out_of_range。
    const ChunkRecord& at(const CacheKey& key) const;
    std::size_t size() const noexcept { return records_.size(); }

    // 在指定层分配槽位并登记副本，返回该副本的位置。该层已有副本时抛
    // logic_error；层满时抛 CapacityError。
    TierLocation place(const CacheKey& key, StorageTier tier,
                       ChunkState state = ChunkState::READY);
    // 移除副本并把槽位还给对应层。该层无副本时抛 logic_error。
    void remove(const CacheKey& key, StorageTier tier);
    // 只改指定层副本的状态，不影响其他层。该层无副本时抛 logic_error。
    void set_state(const CacheKey& key, StorageTier tier, ChunkState state);

    void add_ref(const CacheKey& key);
    void release_ref(const CacheKey& key);
    void pin(const CacheKey& key);
    void unpin(const CacheKey& key);
    void touch(const CacheKey& key);
    // KV 内容变化时递增版本；纯位置迁移不改版本。
    void bump_version(const CacheKey& key);

    std::size_t replica_count(const CacheKey& key) const;

    struct TierStats {
        std::size_t capacity;
        std::size_t available;
        std::size_t allocated_chunks;
        std::size_t used_bytes;
        std::size_t available_bytes;
        // 目录中在该层登记了副本的 chunk 数，与 allocated_chunks 一致。
        std::size_t replicas;
    };
    TierStats stats(StorageTier tier) const;

    // 删除身份。仍存在副本、引用或钉住时抛 logic_error，防止悬空引用。
    void destroy(const CacheKey& key);

private:
    ChunkRecord& lookup(const CacheKey& key);
    std::optional<Replica>& replica_slot(ChunkRecord& record, StorageTier tier);
    Tier& tier_for(StorageTier tier) const;

    Tier* tiers_[3]; // 按 StorageTier 枚举值下标。
    std::unordered_map<CacheKey, ChunkRecord, CacheKeyHash> records_;
};

} // namespace kvflux::v3

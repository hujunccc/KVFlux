#include "kvflux/v3/cache_directory.h"

#include <stdexcept>
#include <utility>

namespace kvflux::v3 {

std::size_t CacheKeyHash::operator()(const CacheKey& key) const noexcept {
    // 只做 map 分桶；相等性由 operator== 逐字段精确比较。
    std::size_t h = std::hash<std::string>{}(key.model_id);
    const auto mix = [&h](std::uint64_t v) {
        h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(key.layout_version);
    mix(static_cast<std::uint64_t>(key.dtype));
    mix(key.first_layer);
    mix(key.last_layer);
    mix(key.prefix_hash);
    return h;
}

CacheKey make_cache_key(std::string model_id, std::uint32_t layout_version, DType dtype,
                        std::uint32_t first_layer, std::uint32_t last_layer,
                        Tokens prefix) {
    if (first_layer >= last_layer) {
        throw std::invalid_argument("layer range must be non-empty");
    }
    CacheKey key;
    key.model_id = std::move(model_id);
    key.layout_version = layout_version;
    key.dtype = dtype;
    key.first_layer = first_layer;
    key.last_layer = last_layer;
    key.prefix_hash = BlockManager::prefix_hash(prefix);
    key.prefix = std::move(prefix);
    return key;
}

CacheDirectory::CacheDirectory(Tier& gpu, Tier& cpu, Tier& nvme) {
    tiers_[static_cast<std::size_t>(StorageTier::GPU)] = &gpu;
    tiers_[static_cast<std::size_t>(StorageTier::CPU)] = &cpu;
    tiers_[static_cast<std::size_t>(StorageTier::NVME)] = &nvme;
    for (std::size_t i = 0; i < 3; ++i) {
        if (tiers_[i]->kind() != static_cast<StorageTier>(i)) {
            throw std::invalid_argument("tier kind does not match its directory slot");
        }
    }
}

void CacheDirectory::create(const CacheKey& key) {
    if (records_.find(key) != records_.end()) {
        throw std::invalid_argument("cache key already registered");
    }
    ChunkRecord record;
    record.key = key;
    record.last_access = std::chrono::steady_clock::now();
    records_.emplace(key, std::move(record));
}

bool CacheDirectory::contains(const CacheKey& key) const {
    return records_.find(key) != records_.end();
}

const ChunkRecord& CacheDirectory::at(const CacheKey& key) const {
    const auto it = records_.find(key);
    if (it == records_.end()) throw std::out_of_range("cache key not registered");
    return it->second;
}

ChunkRecord& CacheDirectory::lookup(const CacheKey& key) {
    const auto it = records_.find(key);
    if (it == records_.end()) throw std::out_of_range("cache key not registered");
    return it->second;
}

Tier& CacheDirectory::tier_for(StorageTier tier) const {
    return *tiers_[static_cast<std::size_t>(tier)];
}

std::optional<Replica>& CacheDirectory::replica_slot(ChunkRecord& record, StorageTier tier) {
    switch (tier) {
        case StorageTier::GPU: return record.gpu;
        case StorageTier::CPU: return record.cpu;
        case StorageTier::NVME: return record.nvme;
    }
    throw std::invalid_argument("unknown storage tier");
}

TierLocation CacheDirectory::place(const CacheKey& key, StorageTier tier, ChunkState state) {
    auto& record = lookup(key);
    auto& slot = replica_slot(record, tier);
    if (slot.has_value()) {
        throw std::logic_error("tier already holds a replica of this chunk");
    }
    auto location = tier_for(tier).allocate();
    slot = Replica{location, state};
    record.last_access = std::chrono::steady_clock::now();
    return location;
}

void CacheDirectory::remove(const CacheKey& key, StorageTier tier) {
    auto& record = lookup(key);
    auto& slot = replica_slot(record, tier);
    if (!slot.has_value()) {
        throw std::logic_error("tier holds no replica of this chunk");
    }
    // 传输进行中的副本不能移除，否则槽位会在数据复制完成前被回收。
    // 失败的副本（ERROR）允许移除以清理现场。
    if (slot->state != ChunkState::READY && slot->state != ChunkState::ERROR) {
        throw std::logic_error("replica is not ready; finish or fail its transfer first");
    }
    tier_for(tier).release(slot->location);
    slot.reset();
    record.last_access = std::chrono::steady_clock::now();
}

void CacheDirectory::set_state(const CacheKey& key, StorageTier tier, ChunkState state) {
    auto& record = lookup(key);
    auto& slot = replica_slot(record, tier);
    if (!slot.has_value()) {
        throw std::logic_error("tier holds no replica of this chunk");
    }
    slot->state = state;
}

void CacheDirectory::add_ref(const CacheKey& key) { ++lookup(key).ref_count; }

void CacheDirectory::release_ref(const CacheKey& key) {
    auto& record = lookup(key);
    if (record.ref_count == 0) throw std::logic_error("reference count already zero");
    --record.ref_count;
}

void CacheDirectory::pin(const CacheKey& key) { ++lookup(key).pin_count; }

void CacheDirectory::unpin(const CacheKey& key) {
    auto& record = lookup(key);
    if (record.pin_count == 0) throw std::logic_error("pin count already zero");
    --record.pin_count;
}

void CacheDirectory::touch(const CacheKey& key) {
    lookup(key).last_access = std::chrono::steady_clock::now();
}

void CacheDirectory::bump_version(const CacheKey& key) { ++lookup(key).version; }

std::size_t CacheDirectory::replica_count(const CacheKey& key) const {
    const auto& record = at(key);
    return static_cast<std::size_t>(record.gpu.has_value()) +
           static_cast<std::size_t>(record.cpu.has_value()) +
           static_cast<std::size_t>(record.nvme.has_value());
}

CacheDirectory::TierStats CacheDirectory::stats(StorageTier tier) const {
    const Tier& t = tier_for(tier);
    std::size_t replicas = 0;
    for (const auto& entry : records_) {
        const auto& record = entry.second;
        const auto& slot =
            tier == StorageTier::GPU ? record.gpu
            : tier == StorageTier::CPU ? record.cpu
                                       : record.nvme;
        if (slot.has_value()) ++replicas;
    }
    return TierStats{t.capacity(), t.available(), t.allocated_chunks(),
                     t.used_bytes(), t.available_bytes(), replicas};
}

void CacheDirectory::destroy(const CacheKey& key) {
    const auto& record = at(key);
    if (record.gpu || record.cpu || record.nvme) {
        throw std::logic_error("chunk still has replicas");
    }
    if (record.ref_count != 0 || record.pin_count != 0) {
        throw std::logic_error("chunk is still referenced or pinned");
    }
    records_.erase(key);
}

} // namespace kvflux::v3

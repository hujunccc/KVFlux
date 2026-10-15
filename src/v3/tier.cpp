#include "kvflux/v3/tier.h"

#include <limits>
#include <stdexcept>

namespace kvflux::v3 {

SlotTier::SlotTier(StorageTier kind, std::size_t chunk_slots, std::size_t chunk_bytes)
    : kind_(kind), chunk_bytes_(chunk_bytes), slots_(chunk_slots) {
    if (chunk_slots == 0 || chunk_bytes == 0) {
        throw std::invalid_argument("chunk slots and chunk bytes must be positive");
    }
    for (std::size_t i = 0; i + 1 < slots_.size(); ++i) slots_[i].next_free = i + 1;
}

TierLocation SlotTier::allocate() {
    if (free_head_ == none) throw CapacityError();
    auto& s = slots_[free_head_];
    const std::size_t id = free_head_;
    free_head_ = s.next_free;
    s.next_free = none;
    s.free = false;
    if (s.generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("slot generation exhausted");
    }
    ++s.generation;
    ++allocated_;
    return {id, s.generation};
}

void SlotTier::release(const TierLocation& location) {
    if (location.slot >= slots_.size() ||
        slots_[location.slot].generation != location.generation) {
        throw std::invalid_argument("invalid or stale tier location");
    }
    auto& s = slots_[location.slot];
    if (s.free) throw std::invalid_argument("tier location already released");
    s.free = true;
    s.next_free = free_head_;
    free_head_ = location.slot;
    --allocated_;
}

GpuTier::GpuTier(PhysicalBlockPool& pool, std::size_t chunk_bytes)
    : pool_(pool), chunk_bytes_(chunk_bytes) {
    if (chunk_bytes == 0) throw std::invalid_argument("chunk bytes must be positive");
}

std::size_t GpuTier::capacity() const noexcept { return pool_.capacity(); }

std::size_t GpuTier::available() const noexcept { return pool_.available(); }

TierLocation GpuTier::allocate() {
    const auto handle = pool_.allocate();
    ++allocated_;
    return {pool_.id(handle), handle.generation};
}

void GpuTier::release(const TierLocation& location) {
    // 旧 generation 或已归还的编号会被池的句柄校验拒绝。
    pool_.free(PhysicalBlockHandle{location.slot, location.generation});
    --allocated_;
}

} // namespace kvflux::v3

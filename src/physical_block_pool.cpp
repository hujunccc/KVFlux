#include "kvflux/physical_block_pool.h"

#include <stdexcept>
#include <utility>

namespace kvflux {

PhysicalBlockPool::PhysicalBlockPool(std::size_t capacity, std::size_t block_size,
                                     BlockManager::Hasher hasher)
    : manager_(capacity, block_size, std::move(hasher)) {}

PhysicalBlockHandle PhysicalBlockPool::allocate() { return manager_.allocate(); }

void PhysicalBlockPool::retain(PhysicalBlockHandle handle) { manager_.retain(handle); }

void PhysicalBlockPool::free(PhysicalBlockHandle handle) { manager_.release(handle); }

void PhysicalBlockPool::publish(PhysicalBlockHandle handle, const Tokens& prefix) {
    manager_.publish(handle, prefix);
}

bool PhysicalBlockPool::lookup(const Tokens& prefix, PhysicalBlockHandle& out) {
    return manager_.lookup(prefix, out);
}

bool PhysicalBlockPool::is_published(PhysicalBlockHandle handle) const {
    return manager_.is_published(handle);
}

PhysicalBlockID PhysicalBlockPool::id(PhysicalBlockHandle handle) const {
    // ref_count 会同时检查范围、是否已经归还、以及 generation 是否匹配。
    // 仅凭 handle.id 访问存储层会绕过这些检查。
    if (manager_.ref_count(handle) == 0) {
        throw std::invalid_argument("physical block has no active reference");
    }
    return handle.id;
}

std::size_t PhysicalBlockPool::ref_count(PhysicalBlockHandle handle) const {
    return manager_.ref_count(handle);
}

std::size_t PhysicalBlockPool::capacity() const noexcept { return manager_.capacity(); }

std::size_t PhysicalBlockPool::block_size() const noexcept { return manager_.block_size(); }

std::size_t PhysicalBlockPool::available() const noexcept {
    const auto stats = manager_.stats();
    return stats.free + stats.cached_idle;
}

} // namespace kvflux

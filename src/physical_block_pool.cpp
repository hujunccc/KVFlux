#include "kvflux/physical_block_pool.h"

#include <stdexcept>

namespace kvflux {

PhysicalBlockPool::PhysicalBlockPool(std::size_t capacity, std::size_t block_size)
    : manager_(capacity, block_size) {}

PhysicalBlockHandle PhysicalBlockPool::allocate() { return manager_.allocate(); }

void PhysicalBlockPool::retain(PhysicalBlockHandle handle) { manager_.retain(handle); }

void PhysicalBlockPool::free(PhysicalBlockHandle handle) { manager_.release(handle); }

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
    // v2 池不发布缓存块，所以所有可用页都在 free list 中。
    return manager_.stats().free;
}

} // namespace kvflux

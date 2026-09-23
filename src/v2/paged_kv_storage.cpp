#include "kvflux/v2/paged_kv_storage.h"

#include <limits>
#include <stdexcept>

namespace kvflux::v2 {
namespace {

std::size_t backing_pages(std::size_t physical_pages) {
    if (physical_pages > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::overflow_error("K/V backing page count overflow");
    }
    return 2 * physical_pages;
}

} // namespace

PagedKVStorage::PagedKVStorage(PhysicalBlockPool& pool, std::size_t num_kv_heads,
                               std::size_t head_size, DType dtype, int device)
    : pool_(pool),
      layout_(pool.capacity(), num_kv_heads, pool.block_size(), head_size, dtype),
      backing_(backing_pages(pool.capacity()), layout_.page_bytes(), pool.block_size(), device) {}

void* PagedKVStorage::key_base() const { return backing_.device_address(BlockId{0}); }

void* PagedKVStorage::value_base() const {
    return backing_.device_address(BlockId{pool_.capacity()});
}

void* PagedKVStorage::page_address(KVKind kind, PhysicalBlockHandle handle) const {
    const auto id = pool_.id(handle);
    const auto offset = layout_.page_byte_offset(kind, id);
    return static_cast<unsigned char*>(key_base()) + offset;
}

void* PagedKVStorage::element_address(KVKind kind, PhysicalBlockHandle handle,
                                      std::size_t head, std::size_t token,
                                      std::size_t dimension) const {
    const auto id = pool_.id(handle);
    const auto offset = layout_.byte_offset(kind, id, head, token, dimension);
    return static_cast<unsigned char*>(key_base()) + offset;
}

void* PagedKVStorage::slot_address(KVKind kind, PhysicalSlot slot,
                                   std::size_t head, std::size_t dimension) const {
    const auto offset = layout_.slot_byte_offset(kind, slot, head, dimension);
    return static_cast<unsigned char*>(key_base()) + offset;
}

} // namespace kvflux::v2

#include "kvflux/v3/host_buffers.h"

#include <limits>
#include <stdexcept>

namespace kvflux::v3 {

HeapChunkPool::HeapChunkPool(std::size_t slots, std::size_t chunk_bytes)
    : slots_(slots), chunk_bytes_(chunk_bytes) {
    if (slots == 0 || chunk_bytes == 0) {
        throw std::invalid_argument("slots and chunk bytes must be positive");
    }
    if (slots > std::numeric_limits<std::size_t>::max() / chunk_bytes) {
        throw std::overflow_error("heap chunk pool byte size overflow");
    }
    bytes_.resize(slots * chunk_bytes);
}

void* HeapChunkPool::data(std::size_t slot) {
    return const_cast<void*>(static_cast<const HeapChunkPool*>(this)->data(slot));
}

const void* HeapChunkPool::data(std::size_t slot) const {
    if (slot >= slots_) throw std::invalid_argument("chunk slot out of range");
    return bytes_.data() + slot * chunk_bytes_;
}

} // namespace kvflux::v3

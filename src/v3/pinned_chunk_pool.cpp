#include "kvflux/v3/pinned_chunk_pool.h"

#include <stdexcept>

namespace kvflux::v3 {

PinnedChunkPool::PinnedChunkPool(std::size_t slots, std::size_t chunk_bytes)
    : chunk_bytes_(chunk_bytes) {
    if (slots == 0 || chunk_bytes == 0) {
        throw std::invalid_argument("slots and chunk bytes must be positive");
    }
    buffers_.reserve(slots);
    for (std::size_t i = 0; i < slots; ++i) {
        buffers_.push_back(std::make_unique<PinnedBuffer>(chunk_bytes));
    }
}

void* PinnedChunkPool::data(std::size_t slot) {
    return const_cast<void*>(static_cast<const PinnedChunkPool*>(this)->data(slot));
}

const void* PinnedChunkPool::data(std::size_t slot) const {
    if (slot >= buffers_.size()) throw std::invalid_argument("chunk slot out of range");
    return buffers_[slot]->data();
}

} // namespace kvflux::v3

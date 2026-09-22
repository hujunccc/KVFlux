#pragma once
#include "kvflux/pinned_memory.h"
#include "cuda_utils.h"

namespace kvflux {
struct PinnedBuffer::Storage {
    explicit Storage(std::size_t size) : bytes(size) {
        if (!size) throw std::invalid_argument("pinned buffer size must be positive");
        detail::check(cudaHostAlloc(&pointer, size, cudaHostAllocPortable), "cudaHostAlloc");
    }
    ~Storage() noexcept { (void)cudaFreeHost(pointer); }
    void* pointer = nullptr;
    std::size_t bytes;
    bool busy = false;
};
} // namespace kvflux

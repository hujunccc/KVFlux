#pragma once

#include "kvflux/gpu_memory_pool.h"
#include "kvflux/v3/tier.h"

#include <cstddef>

namespace kvflux::v3 {

// GPU 侧 chunk 的真实显存。V2 PhysicalBlockPool 只发编号，本类复用 v1
// GpuMemoryPool 的一次性 cudaMalloc 和地址映射，slot 即 BlockId。
// 槽位生命周期由 GpuTier/CacheDirectory 管理，本类只做地址换算。
class GpuChunkStore {
public:
    GpuChunkStore(std::size_t total_slots, std::size_t chunk_bytes, int device = 0);

    // 主机不能解引用返回值；只检查槽位范围，不检查 generation。
    void* device_ptr(const TierLocation& location) const;
    void* device_ptr(std::size_t slot) const;
    std::size_t chunk_bytes() const noexcept { return pool_.block_bytes(); }
    std::size_t slots() const noexcept { return pool_.total_blocks(); }
    int device() const noexcept { return pool_.device(); }

private:
    GpuMemoryPool pool_;
};

} // namespace kvflux::v3

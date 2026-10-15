#include "kvflux/v3/gpu_chunk_store.h"

namespace kvflux::v3 {

GpuChunkStore::GpuChunkStore(std::size_t total_slots, std::size_t chunk_bytes, int device)
    : pool_(total_slots, chunk_bytes, 1, device) {}

void* GpuChunkStore::device_ptr(const TierLocation& location) const {
    return device_ptr(location.slot);
}

void* GpuChunkStore::device_ptr(std::size_t slot) const {
    // device_address 检查 id 范围；generation 校验属于 GpuTier。
    return pool_.device_address(slot);
}

} // namespace kvflux::v3

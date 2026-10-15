#pragma once

#include "kvflux/pinned_memory.h"
#include "kvflux/v3/host_buffers.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace kvflux::v3 {

// 复用 v1 的 PinnedBuffer：每个 CPU tier 槽位对应一个长期持有的
// page-locked buffer，分配不发生在迁移热路径上。
class PinnedChunkPool : public HostChunkBuffers {
public:
    PinnedChunkPool(std::size_t slots, std::size_t chunk_bytes);

    std::size_t chunk_bytes() const noexcept override { return chunk_bytes_; }
    std::size_t slots() const noexcept override { return buffers_.size(); }
    void* data(std::size_t slot) override;
    const void* data(std::size_t slot) const override;

private:
    std::size_t chunk_bytes_;
    std::vector<std::unique_ptr<PinnedBuffer>> buffers_;
};

} // namespace kvflux::v3

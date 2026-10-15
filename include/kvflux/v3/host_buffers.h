#pragma once

#include "kvflux/v3/tier.h"

#include <cstddef>
#include <vector>

namespace kvflux::v3 {

// CPU 侧 chunk 字节的宿主内存访问。Tier 管槽位元数据，本接口管数据；
// 槽位生命周期由 Tier/CacheDirectory 决定，这里只做范围检查。
class HostChunkBuffers {
public:
    virtual ~HostChunkBuffers() = default;

    virtual std::size_t chunk_bytes() const noexcept = 0;
    virtual std::size_t slots() const noexcept = 0;
    // 越界槽位抛 invalid_argument；generation 校验属于 Tier。
    virtual void* data(std::size_t slot) = 0;
    virtual const void* data(std::size_t slot) const = 0;
};

// 普通堆内存版本：无 CUDA 依赖，供 CPU-only 路径和测试使用。
class HeapChunkPool : public HostChunkBuffers {
public:
    HeapChunkPool(std::size_t slots, std::size_t chunk_bytes);

    std::size_t chunk_bytes() const noexcept override { return chunk_bytes_; }
    std::size_t slots() const noexcept override { return slots_; }
    void* data(std::size_t slot) override;
    const void* data(std::size_t slot) const override;

private:
    std::size_t slots_;
    std::size_t chunk_bytes_;
    std::vector<unsigned char> bytes_;
};

} // namespace kvflux::v3

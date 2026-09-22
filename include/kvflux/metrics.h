#pragma once
#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace kvflux {

// 只统计成功完成的 batch。device_ms 为同一 transfer stream 上 CUDA event 区间之和。
struct TransferDirectionMetrics {
    std::uint64_t bytes = 0, batches = 0;
    double device_ms = 0;
    double latency_us() const noexcept { return batches ? device_ms * 1000 / batches : 0; }
    double bandwidth_gbps() const noexcept { return device_ms > 0 ? bytes / device_ms / 1e6 : 0; }
};
struct TransferMetrics {
    TransferDirectionMetrics gpu_to_cpu, cpu_to_gpu;
    std::uint64_t failed_batches = 0;
};

struct CacheMetrics {
    std::size_t gpu_blocks_total = 0, gpu_blocks_used = 0, gpu_blocks_free = 0;
    std::size_t cpu_cached_blocks = 0, logical_blocks = 0, transferring_blocks = 0;
    TransferMetrics transfers;
    std::uint64_t offload_count = 0, load_count = 0;
    std::uint64_t prefetch_count = 0, prefetch_skipped = 0;
    // hit: 首次 demand 已由预取准备好；miss: demand 到来时数据尚不可用。
    // 普通 GPU resident hit 不属于这两项；late 是 miss 中已在预取的部分。
    std::uint64_t prefetch_hits = 0, prefetch_misses = 0, prefetch_late = 0;
    std::uint64_t prefetch_unused = 0;
    std::size_t prefetch_outstanding = 0; // 尚未消费的预取，含就绪和排队项。
    std::uint64_t demand_count = 0, gpu_demand_hits = 0;
    double request_stall_ms = 0;
};

// 稳定的 key=value 文本，计数器自对象创建起累计；不隐式同步或重置。
std::ostream& operator<<(std::ostream& out, const CacheMetrics& metrics);
} // namespace kvflux

#pragma once
#include "kvflux/cuda_stream.h"
#include "kvflux/gpu_memory_pool.h"
#include "kvflux/pinned_memory.h"
#include "kvflux/metrics.h"
#include <memory>
#include <vector>

namespace kvflux {

// Runtime 必须先于 pool 析构（声明顺序：pool，再 runtime）。单 host 线程使用。
// 一个 batch 对应一个连续 host buffer，GPU block id 可以不连续。
class AsyncTransferRuntime {
public:
    explicit AsyncTransferRuntime(GpuMemoryPool& pool);
    ~AsyncTransferRuntime() noexcept;
    AsyncTransferRuntime(const AsyncTransferRuntime&) = delete;
    AsyncTransferRuntime& operator=(const AsyncTransferRuntime&) = delete;
    void write_batch(const std::vector<BlockHandle>& blocks, PinnedBuffer& source);
    void read_batch(const std::vector<BlockHandle>& blocks, PinnedBuffer& destination);
    void write_block(BlockHandle block, PinnedBuffer& source) { write_batch({block}, source); }
    void read_block(BlockHandle block, PinnedBuffer& destination) { read_batch({block}, destination); }
    // 统一等待，之后更新初始化状态并归还运行时引用，不逐块 host 等待。
    void synchronize();
    std::size_t pending_batches() const noexcept { return pending_.size(); }
    TransferMetrics metrics() const noexcept { return metrics_; }
    const CudaStream& compute_stream() const noexcept { return compute_; }
    const CudaStream& transfer_stream() const noexcept { return transfer_; }
    // GPU 侧依赖：让随后提交到 compute stream 的任务等待当前已提交的传输。
    // 不阻塞 CPU；pool 的 host 侧就绪状态仍在 synchronize() 后更新。
    void compute_wait_for_transfers();
private:
    struct Pending;
    void submit(const std::vector<BlockHandle>& blocks, PinnedBuffer& buffer, bool write);
    void finish(bool success) noexcept;
    GpuMemoryPool& pool_;
    CudaStream compute_, transfer_;
    std::vector<std::unique_ptr<Pending>> pending_;
    // 复用完成 batch 的 CUDA event，避免 steady state 每次迁移创建/销毁 event。
    std::vector<std::unique_ptr<Pending>> recycled_;
    TransferMetrics metrics_;
};
} // namespace kvflux

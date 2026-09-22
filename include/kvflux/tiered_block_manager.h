#pragma once
#include "kvflux/async_transfer.h"
#include "kvflux/detail/index_lru.h"
#include <optional>

namespace kvflux {

// 逻辑 id 不再等于 GPU 槽位。用独立类型避免误传给 GpuMemoryPool。
struct LogicalBlockHandle {
    std::size_t id;
    std::uint64_t generation;
    bool operator==(const LogicalBlockHandle& other) const noexcept {
        return id == other.id && generation == other.generation;
    }
};
enum class BlockLocation { GPU_RESIDENT, CPU_RESIDENT, TRANSFERRING };

class TieredBlockManager {
public:
    // 单 host 线程调用；句柄仅用于创建它的 manager。manager 必须比所有 lease 活得更久。
    // CPU 为每个逻辑槽位预分配一个 pinned backing buffer（包括 GPU resident 块）。
    // 总唯一块数上限是 cpu_blocks；物理 GPU 槽位上限是 gpu_blocks。
    TieredBlockManager(std::size_t gpu_blocks, std::size_t cpu_blocks,
                      std::size_t block_bytes, std::size_t tokens_per_block, int device = 0);
    ~TieredBlockManager() noexcept;
    TieredBlockManager(const TieredBlockManager&) = delete;
    TieredBlockManager& operator=(const TieredBlockManager&) = delete;

    class GpuLease {
    public:
        ~GpuLease() noexcept;
        GpuLease(const GpuLease&) = delete;
        GpuLease& operator=(const GpuLease&) = delete;
        GpuLease(GpuLease&& other) noexcept;
        GpuLease& operator=(GpuLease&&) = delete;
        void* data() const noexcept { return address_; }
    private:
        friend class TieredBlockManager;
        GpuLease(TieredBlockManager* owner, LogicalBlockHandle h, void* address)
            : owner_(owner), handle_(h), address_(address) {}
        TieredBlockManager* owner_;
        LogicalBlockHandle handle_;
        void* address_;
    };
    // 创建一个完整的数据块；在返回之前写入 GPU。GPU 空间不足时同步 offload LRU。
    LogicalBlockHandle create(const void* host, std::size_t bytes);
    void retain(LogicalBlockHandle h);
    void release(LogicalBlockHandle h);
    BlockLocation location(LogicalBlockHandle h) const;
    std::size_t ref_count(LogicalBlockHandle h) const;
    // lease 存活期间禁止迁移；调用方必须在 kernel 完成后才销毁 lease。
    // 返回时数据已在 GPU；仅持有逻辑引用不能保护先前取得的 GPU 地址。
    GpuLease acquire_gpu(LogicalBlockHandle h);
    void read_block(LogicalBlockHandle h, void* host, std::size_t bytes);

    // 提交迁移。offload 不等待；load 在 GPU 满时可能等待 LRU 搬出以取得槽位。
    void offload(LogicalBlockHandle h);
    void load(LogicalBlockHandle h);
    void wait();
    // 非阻塞推进：回收已完成迁移，并提交等待空闲槽位的预取。
    bool poll();
    // 预取 order[current+1 ... current+next_n]，返回新接纳数（含排队项）。
    // 保护当前块及整个窗口；不足可迁移槽位则跳过。用 poll()/wait() 推进排队项。
    std::size_t prefetch_next(const std::vector<LogicalBlockHandle>& order,
                              std::size_t current, std::size_t next_n);
    struct Stats {
        std::size_t logical_blocks, gpu_resident, cpu_resident, transferring;
        std::size_t gpu_free, gpu_capacity, cpu_capacity, cpu_reserved_bytes;
        std::size_t offloads, loads, prefetch_submitted, prefetch_skipped;
    };
    Stats stats() const noexcept;
    CacheMetrics metrics() const noexcept;
    std::size_t block_bytes() const noexcept { return gpu_.block_bytes(); }
    int device() const noexcept { return gpu_.device(); }
    const CudaStream& compute_stream() const noexcept { return transfers_.compute_stream(); }
private:
    enum class Move { None, ToCpu, ToGpu, QueuedLoad };
    struct Entry {
        bool live = false;
        LogicalBlockHandle handle{};
        BlockLocation location = BlockLocation::CPU_RESIDENT;
        Move move = Move::None;
        std::optional<BlockHandle> gpu;
        std::size_t pins = 0;
        bool prefetched = false;
    };
    Entry& checked(LogicalBlockHandle h);
    const Entry& checked(LogicalBlockHandle h) const;
    void unpin(LogicalBlockHandle h) noexcept;
    void discard_prefetch(Entry& entry) noexcept;
    void start_load(LogicalBlockHandle h);
    void start_offload(LogicalBlockHandle h);
    void finish(bool success) noexcept;
    void dispatch_queued();
    void make_room();
    std::size_t victim(const std::vector<bool>& protected_ids) const noexcept;
    static BlockHandle raw(LogicalBlockHandle h) { return {h.id, h.generation}; }

    BlockManager logical_;
    GpuMemoryPool gpu_;
    AsyncTransferRuntime transfers_;
    std::vector<std::unique_ptr<PinnedBuffer>> backing_;
    std::vector<Entry> entries_;
    detail::IndexLru resident_lru_;
    std::vector<LogicalBlockHandle> pending_, queued_, dispatch_work_;
    std::size_t cpu_reserved_bytes_;
    std::size_t offloads_ = 0, loads_ = 0, prefetch_submitted_ = 0, prefetch_skipped_ = 0;
    std::uint64_t prefetch_hits_ = 0, prefetch_misses_ = 0, prefetch_late_ = 0, prefetch_unused_ = 0;
    std::uint64_t demand_count_ = 0, gpu_demand_hits_ = 0;
    double request_stall_ms_ = 0;
};
} // namespace kvflux

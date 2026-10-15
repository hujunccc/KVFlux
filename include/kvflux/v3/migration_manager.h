#pragma once

#include "kvflux/cuda_stream.h"
#include "kvflux/v3/cache_directory.h"
#include "kvflux/v3/chunk_file_store.h"
#include "kvflux/v3/gpu_chunk_store.h"
#include "kvflux/v3/host_buffers.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace kvflux::v3 {

enum class MigrationState { QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED };

using TaskID = std::uint64_t;

struct MigrationTaskInfo {
    TaskID id;
    CacheKey key;
    StorageTier source;
    StorageTier destination;
    MigrationState state;
    int priority;
    std::string error;
};

// 异步迁移引擎。接受迁移任务并执行，不决定缓存淘汰策略：完成后源副本
// 回到 READY，是否移除由调用方决定。
//
// 并发模型：submit/poll/wait/cancel 只能由同一个 owner 线程调用，
// CacheDirectory、Tier 的全部变更都发生在 owner 线程（在 poll/wait 的
// finalize 里）。NVMe I/O 由内部线程池执行，worker 只读写文件和早已
// 分配好的 CPU 槽位内存，绝不触碰目录。GPU 传输在 submit 时提交到
// 独立 transfer stream，用 CUDA event 跟踪，与 compute stream 的计算
// 互不阻塞。
//
// 安全：任务期间 chunk 被 pin、源/目标副本都不是 READY（目录拒绝移除），
// TierLocation 带 generation，过期任务无法落到重用的槽位上。CPU pinned
// 槽位总数固定，提交时容量不足直接抛 CapacityError，不会随任务堆积
// 无限增长。cancel 对 QUEUED 任务立即生效；对已提交到设备的 DMA 只
// 记录取消请求，等 event 确认完成后才回收缓冲区。
class MigrationManager {
public:
    // 所有引用对象必须比本类活得久；io_threads 是 NVMe I/O 线程数。
    MigrationManager(CacheDirectory& directory, GpuChunkStore& gpu_store,
                     HostChunkBuffers& cpu_buffers, ChunkFileStore& store,
                     std::size_t io_threads = 4);
    ~MigrationManager() noexcept;
    MigrationManager(const MigrationManager&) = delete;
    MigrationManager& operator=(const MigrationManager&) = delete;

    // 方向由目录推断：可直达 destination 的 READY 副本中恰好一个作为源
    //（NVMe↔GPU 无直达路径，这样的副本不参与推断），destination 不能有副本。
    // NVMe 与 GPU 之间必须经 CPU 中转。CPU→NVMe 需要 valid_tokens 写入文件头。
    // 目标层容量不足抛 CapacityError，任务不创建。
    TaskID submit(const CacheKey& key, StorageTier destination,
                  std::uint64_t valid_tokens = 0, int priority = 0);

    // 非阻塞推进并返回当前状态；任务到终态后仍保持可查。
    MigrationState poll(TaskID id);
    // 阻塞直到终态（COMPLETED/FAILED/CANCELLED）。
    MigrationState wait(TaskID id);
    // QUEUED → 立即 CANCELLED；RUNNING → 记录取消，底层操作安全完成后
    // 以 CANCELLED 收尾（目标副本被丢弃而非标 READY）。已在终态返回 false。
    bool cancel(TaskID id);

    MigrationTaskInfo info(TaskID id) const;
    std::size_t pending_count() const;

private:
    struct Task {
        TaskID id = 0;
        CacheKey key;
        StorageTier source = StorageTier::GPU;
        StorageTier destination = StorageTier::CPU;
        int priority = 0;
        TierLocation source_loc{};
        TierLocation destination_loc{};
        ChunkFileMeta meta;                  // NVMe 写入头 / 读取校验
        // 以下跨线程字段：worker 只写 io_result/error，其余只有 owner 线程碰。
        std::atomic<MigrationState> state{MigrationState::QUEUED};
        std::atomic<bool> cancel_requested{false};
        std::atomic<int> io_result{0};       // 0 未完成，1 成功，-1 失败
        std::string error;
        cudaEvent_t done = nullptr;          // GPU 传输的完成事件
        bool uses_cuda() const { return source == StorageTier::GPU ||
                                        destination == StorageTier::GPU; }
    };

    void finalize(Task& task);
    void finalize_cancelled(Task& task);
    void rollback_submit(Task& task);
    void worker_loop();
    void do_io(Task& task);
    Task& find(TaskID id) const;

    CacheDirectory& directory_;
    GpuChunkStore& gpu_store_;
    HostChunkBuffers& cpu_buffers_;
    ChunkFileStore& store_;
    CudaStream transfer_stream_;

    TaskID next_id_ = 1;
    std::unordered_map<TaskID, std::shared_ptr<Task>> tasks_; // owner 线程

    mutable std::mutex io_mu_;
    std::condition_variable io_work_cv_;
    std::condition_variable io_done_cv_;
    std::deque<std::shared_ptr<Task>> io_queue_; // 受 io_mu_ 保护
    bool io_stop_ = false;
    std::vector<std::thread> workers_;
};

} // namespace kvflux::v3

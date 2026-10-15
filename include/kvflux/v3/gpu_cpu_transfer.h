#pragma once

#include "kvflux/cuda_stream.h"
#include "kvflux/v3/cache_directory.h"
#include "kvflux/v3/gpu_chunk_store.h"
#include "kvflux/v3/host_buffers.h"

#include <cuda_runtime_api.h>

#include <unordered_map>

namespace kvflux::v3 {

// GPU ↔ CPU 的 chunk 迁移编排。传输期间源副本与目标副本都不是 READY，
// 目录因此拒绝提前移除；chunk 被 pin 住，identity 也不能销毁。
// 传输在独立 non-blocking stream 上用 cudaMemcpyAsync 发起，
// finish_* 等待该 chunk 的 CUDA event 后再更新目录状态。
// 与目录一样单线程使用。
class GpuCpuTransfer {
public:
    // directory 管理的 GPU/CPU tier 必须与两个存储的槽位布局一致；
    // 所有引用对象必须比本类活得久。
    GpuCpuTransfer(CacheDirectory& directory, GpuChunkStore& gpu_store,
                   HostChunkBuffers& cpu_buffers, int device = 0);
    ~GpuCpuTransfer() noexcept;
    GpuCpuTransfer(const GpuCpuTransfer&) = delete;
    GpuCpuTransfer& operator=(const GpuCpuTransfer&) = delete;

    // GPU → CPU。要求 GPU 副本 READY 且无 CPU 副本；发起后 GPU 副本与
    // 新 CPU 副本都是 COPYING，chunk 被 pin。源数据在 finish 前不可改动。
    void begin_offload(const CacheKey& key);
    // 等待传输完成：CPU 副本标记 READY；release_gpu 为 true 时释放 GPU
    // 副本（此时不存在未完成的 GPU 使用），否则 GPU 副本回到 READY。
    void finish_offload(const CacheKey& key, bool release_gpu);

    // CPU → GPU。要求 CPU 副本 READY 且无 GPU 副本。
    void begin_onload(const CacheKey& key);
    void finish_onload(const CacheKey& key, bool release_cpu);

    bool has_pending(const CacheKey& key) const;

private:
    struct Pending {
        TierLocation source;
        TierLocation destination;
        bool offload;
        cudaEvent_t done = nullptr;
    };

    void finish(const CacheKey& key, bool offload, bool release_source);

    CacheDirectory& directory_;
    GpuChunkStore& gpu_store_;
    HostChunkBuffers& cpu_buffers_;
    CudaStream stream_;
    std::unordered_map<CacheKey, Pending, CacheKeyHash> pending_;
};

} // namespace kvflux::v3

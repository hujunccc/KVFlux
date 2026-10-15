#pragma once

#include "kvflux/v3/chunk_file_store.h"
#include "kvflux/v3/host_buffers.h"

namespace kvflux::v3 {

// CPU ↔ NVMe 的 chunk 迁移编排。同步执行（V3 允许迁移仍是同步的）；
// 状态流转与 GpuCpuTransfer 一致：源/目标副本在传输期间非 READY，
// 失败时目标副本标记 ERROR 后移除，源副本恢复 READY。
class NvmeTransfer {
public:
    // cpu_buffers 提供 CPU tier 槽位对应的宿主内存；三者必须比本类活得久。
    NvmeTransfer(CacheDirectory& directory, ChunkFileStore& store,
                 HostChunkBuffers& cpu_buffers);

    // CPU READY 副本 → NVMe 文件。登记 NVMe 副本（COPYING）后写入：
    // serialize header → 临时文件 → payload → fsync → 原子 rename → READY。
    // release_cpu 为 true 时成功后移除 CPU 副本。
    void offload(const CacheKey& key, std::uint64_t valid_tokens, bool release_cpu);

    // NVMe READY 副本 → CPU 槽位。读 header → 校验元数据与校验和 →
    // 读 payload → CPU 副本 READY。文件不存在抛 ChunkMissError，损坏或
    // 身份不匹配抛 ChunkCorruptError；失败时 CPU 副本清理回滚。
    // release_nvme 为 true 时成功后删除文件并移除 NVMe 副本。
    void onload(const CacheKey& key, bool release_nvme);

private:
    CacheDirectory& directory_;
    ChunkFileStore& store_;
    HostChunkBuffers& cpu_buffers_;
};

} // namespace kvflux::v3

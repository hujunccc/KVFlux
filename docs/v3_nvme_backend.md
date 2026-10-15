# v3 Milestone 2：NVMe Storage Backend

V3.2 把第三级存储落地为**基于文件的 Chunk Store**，三级缓存闭环打通：
GPU → CPU → NVMe → CPU → GPU 字节完全一致。迁移仍是同步的。

## 文件格式（ChunkFileFormat）

```text
root/chunks/chunk_000024.bin      已发布的完整 chunk
root/chunks/chunk_000024.bin.tmp  写入中的临时文件，永不被当作缓存
```

头部（FileMetadata）按序包含：magic `KVFX`、format version、model id、
layout version、dtype、层范围、prefix hash + prefix 原文、valid token
count、payload size、payload checksum、header checksum。payload 是
**来自 GPU 的真实 K/V tensor 原始字节**，不是 token id 或请求元数据。
ChecksumVerifier 是 FNV-1a 64 位；头部与 payload 各自独立校验。

## 写入与读取

写入（`ChunkFileStore::write`）：serialize header → 临时文件 →
payload → fsync → close → **原子 rename** 发布 → fsync 目录。
崩溃只残留 `.tmp` 文件，`exists`/`read` 永远看不到半成品。

读取（`ChunkFileStore::read`）：读 header → 校验 magic / 版本 /
header checksum → **逐项精确比较身份字段**（错误模型、布局版本、
dtype、层范围、前缀都不能加载）→ 读 payload → 校验 checksum。
文件不存在返回 false（cache miss），损坏抛 `ChunkCorruptError`。

配额模拟磁盘容量上限：超出抛 `ChunkQuotaError`；占用量构造时从磁盘
扫描恢复。真实 ENOSPC 由 `ChunkStoreError` 透传。大量小文件的文件系统
开销、Segmented File、io_uring/SPDK/GPUDirect 均属后续优化。

## 迁移编排（NvmeTransfer）

`NvmeTransfer` 复用 V3.1 的状态机约定：源副本 COPYING、目标副本
COPYING/LOADING，完成后 READY，失败时目标标记 ERROR 移除、源恢复
READY。`onload` 时文件缺失抛 `ChunkMissError` 并回滚 CPU 副本；
`release_source` 时删除文件并归还槽位。CPU 侧内存通过
`HostChunkBuffers` 接口访问，pinned 池（GPU 通路）与堆内存
（CPU-only 测试）可以互换。

## 测试与基准

`kvflux_v3_nvme_chunk_store_tests`（CPU-only）：写入/读取正确、
payload 与 header 损坏检测、文件缺失为 miss 且目录回滚、配额不足
正确失败、错误版本/模型被拒、`.tmp` 半成品不算缓存、配额占用重启后
从磁盘恢复。

`kvflux_v3_three_tier_tests`（GPU）：7.5 的完整闭环——GPU 物理页被
其他 chunk 占用后，A 从 NVMe 恢复到**不同的物理页**，字节完全一致。

```bash
ctest --test-dir build-v3-cuda --output-on-failure -R 'v3'
./build-v3-cuda/benchmark/kvflux_v3_nvme_benchmark \
  benchmark/results/rtx3060-v3/nvme_io_benchmark.csv
```

本机实测（含 fsync）：64 KiB 写约 887 MB/s，16 MiB 写约 812 MB/s、
读约 870 MB/s。CSV 列：`transfer_size_bytes, write_latency_ms,
read_latency_ms, write_bandwidth_mbps, read_bandwidth_mbps`。

接口见 [chunk_file_store.h](../include/kvflux/v3/chunk_file_store.h) 与
[nvme_transfer.h](../include/kvflux/v3/nvme_transfer.h)。

# v3 Milestone 1：GPU ↔ CPU Offloading

V3.1 是 V3 第一个真实数据迁移阶段：KV chunk 可以离开 GPU 显存进入 CPU
pinned memory，并在**新的 GPU 物理页**上正确恢复。不涉及 NVMe。

## 数据面组件

| 组件 | 作用 |
| --- | --- |
| `GpuChunkStore` | 复用 v1 `GpuMemoryPool`：一次 `cudaMalloc`，slot 即 BlockId |
| `PinnedChunkPool` | 每个 CPU tier 槽位对应一个长期 `PinnedBuffer`，实现 `HostChunkBuffers` |
| `HeapChunkPool` | 无 CUDA 的堆内存实现，供 CPU-only 路径与测试使用 |
| `GpuCpuTransfer` | 迁移编排：目录状态机 + `cudaMemcpyAsync` + CUDA event |

槽位元数据（分配、generation、容量统计）仍在 V3.0 的 `Tier` /
`CacheDirectory`；数据面只做地址换算与字节搬运，不改变所有权语义。

## GPU → CPU 流程

```text
begin_offload(key)
  1. 要求 GPU 副本 READY 且无 CPU 副本（否则 logic_error）
  2. GPU 副本 → COPYING，place CPU 副本（COPYING），chunk pin+1
  3. cudaMemcpyAsync D2H 到独立 non-blocking stream，记录 event
finish_offload(key, release_gpu)
  4. cudaEventSynchronize 等待该 chunk 的传输完成
  5. CPU 副本 → READY，unpin
  6. release_gpu 时移除 GPU 副本，物理页立即可被其他 chunk 复用
```

CPU → GPU（`begin_onload` / `finish_onload`）对称，目标副本状态用 LOADING。

## 安全不变量

- **传输未完成不能提前释放**：在途的源/目标副本都不是 READY；V3.1 起
  `CacheDirectory::remove` 只接受 READY 或 ERROR 的副本，COPYING/LOADING/
  EVICTING 一律抛 `logic_error`。chunk 被 pin，`destroy` 也被拒绝。
- **物理页复用不影响逻辑身份**：offload 释放 Block 17 后它可以被别的
  chunk 占用；onload 落到 Block 91，CacheKey 不变、字节不变，
  旧 TierLocation 因 generation 失配被拒绝。
- 发起阶段失败时回滚：目标副本标记 ERROR 后移除，源副本恢复 READY，
  event 销毁，不产生半登记状态。

## 测试与基准

`kvflux_v3_gpu_cpu_migration_tests`（无 GPU 时返回 77 跳过）覆盖：
GPU→CPU、CPU→GPU、GPU→CPU→GPU 回环、重映射后字节一致、原 block 安全
复用且旧位置被拒、传输未完成的释放/销毁/重复发起全部被拒、4 个 chunk
并发在途、FP16/BF16 随机原始字节逐位一致。

```bash
cmake -S . -B build-v3-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-v3-cuda -j 4
ctest --test-dir build-v3-cuda --output-on-failure -R 'v3'
```

带宽基准（pinned ↔ GPU，CUDA event 计时，10 样本中位数）：

```bash
./build-v3-cuda/benchmark/kvflux_v3_bandwidth_benchmark \
  benchmark/results/rtx3060-v3/gpu_cpu_bandwidth.csv
```

RTX 3060 Laptop 实测：大块（16–64 MiB）双向约 **6.2–6.3 GB/s**，
小传输受固定延迟主导（256 KiB 约 25–30 µs）。CSV 列：
`transfer_size_bytes, h2d_latency_ms, d2h_latency_ms,
h2d_bandwidth_gbps, d2h_bandwidth_gbps`。

接口见 [gpu_cpu_transfer.h](../include/kvflux/v3/gpu_cpu_transfer.h)、
[gpu_chunk_store.h](../include/kvflux/v3/gpu_chunk_store.h) 与
[pinned_chunk_pool.h](../include/kvflux/v3/pinned_chunk_pool.h)。

注意：本机 nvcc 12.4 的 host 编译器是 gcc-13，其隐式链接目录会让旧
libstdc++ 盖住 gcc-15 的库，Release 链接报 `__cxa_call_terminate`
未定义；配置时加
`-DCMAKE_EXE_LINKER_FLAGS=-L/usr/lib/gcc/x86_64-linux-gnu/15` 即可。

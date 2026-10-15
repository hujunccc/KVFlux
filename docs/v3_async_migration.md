# v3 Milestone 3：Asynchronous Migration Engine

V3.3 把迁移从同步调用升级为**异步任务引擎**：`MigrationManager` 接受
submit/poll/wait/cancel，GPU↔CPU 走独立 transfer stream 上的
cudaMemcpyAsync + CUDA event，NVMe 读写走内部线程池 + pread/pwrite
（io_uring 属后续优化）。迁移与 attention 可以安全地并发执行。

## 任务模型

`MigrationTaskInfo{ id, key, source, destination, state, priority, error }`，
状态机：QUEUED → RUNNING → COMPLETED；RUNNING → FAILED；
QUEUED/RUNNING → CANCELLED。GPU↔CPU 任务在 submit 时立即提交到设备
（直接 RUNNING）；NVMe 任务进优先级队列（数值大者优先，平级按提交
顺序），由 I/O 线程取走后转 RUNNING。

## 并发模型：owner 线程 + 纯 I/O worker

submit/poll/wait/cancel 只能由同一个 owner 线程调用。**CacheDirectory
和 Tier 的全部变更都发生在 owner 线程**（完成收尾在 poll/wait 的
finalize 里）；NVMe worker 只做文件读写，落盘目标是早已分配好的 CPU
槽位内存，绝不触碰目录——V3.0 起"目录单线程"的约定原样保留。

安全机制：

- 任务期间 chunk 被 **pin**，源/目标副本均非 READY，目录拒绝 remove；
- `TierLocation` 带 **generation**，任务若熬到槽位被释放重用，提交时已
  解析好的旧位置不会被误认（release 需要整个 handle 匹配）；
- CPU pinned 槽位总数固定，容量不足在 **submit 时**抛 CapacityError，
  任务不会排队堆积、无限占用 pinned memory；
- cancel 对 QUEUED 任务立即生效（摘除队列、回收目标槽位）；对已在
  设备上执行的 DMA **无法中途叫停**，只记录取消请求，等 event 确认
  完成后才回收缓冲区，目标副本以丢弃收尾而不是标 READY；
- 复制尚未由 event / I/O 结果确认完成前，目标副本保持 COPYING/LOADING，
  绝不提前发布 READY。

## 验收：100 任务 × 并发 attention

`kvflux_v3_async_migration_tests`（GPU）：预置 100 个 chunk 到 NVMe，
主循环流水线式提交 NVMe→CPU→GPU 共 200 个任务（CPU 槽满即
CapacityError 退避），每 3 轮迭代在同一 `PhysicalBlockPool` 上执行一次
真实 v2 paged attention。结束后逐项验证：GPU 字节与预置数据逐位一致、
attention 输出与迁移前逐位一致、BlockTable 物理页映射不变、池水位复原、
无泄漏无重复释放、全部 chunk pin_count 归零。另有取消排队任务、取消
运行中 CUDA 任务（缓冲区安全回收）、失败任务资源释放、pinned 内存
有界性、完成前不发布 READY 共五个小场景。

## 真实重叠测量

`kvflux_v3_overlap_benchmark` 不是"调用了异步 API"就算重叠：先分别测
纯计算与纯迁移耗时，再对比串行与重叠总耗时。计算量是校准过的
（kernel 数 ≈ migration_ms / kernel_ms）。

```bash
ctest --test-dir build-v3-cuda --output-on-failure -R 'v3_async'
./build-v3-cuda/benchmark/kvflux_v3_overlap_benchmark \
  benchmark/results/rtx3060-v3/async_overlap.csv
```

本机实测（RTX 3060 Laptop，8 × 16 MiB chunk）：compute 20.3 ms +
migration 22.7 ms，串行 43.0 ms，重叠 21.8 ms（≈ 两者较大者），
speedup 1.97x——kernel 与 DMA 确实在不同 stream 上并行。CSV 列：
`chunk_size_bytes, num_chunks, num_kernels, compute_ms, migration_ms,
sequential_ms, overlapped_ms, speedup`。

接口见 [migration_manager.h](../include/kvflux/v3/migration_manager.h)。

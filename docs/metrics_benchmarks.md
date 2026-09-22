# v1.9–v1.10：Metrics 与 Benchmark Suite

公共类型在 [metrics.h](../include/kvflux/metrics.h)。`AsyncTransferRuntime::metrics()` 提供双向迁移统计，`TieredBlockManager::metrics()` 将它与容量、offload、prefetch 和 demand 统计合并。返回的是值快照；读取不会推进迁移或等待 GPU，也不会清零。计数器从对象创建开始累积，实验用计时前后快照相减，gauges 使用结束时值。

```cpp
// manager 为 TieredBlockManager；输出可用于日志或重定向保存。
manager.wait();
std::cout << manager.metrics();
```

[tiered_cache.cpp](../examples/tiered_cache.cpp) 输出全部 metrics。`operator<<` 使用稳定的 `key=value` 字段名；程序可直接访问结构体字段，基准导出 CSV。

## 指标口径

| 输出字段 | 含义 |
| --- | --- |
| `gpu_blocks_total` | GPU 物理槽位总数 |
| `gpu_blocks_used` | 已占用槽位，包括 H2D/D2H 尚未完成的槽位 |
| `gpu_blocks_free` | 可分配槽位；total = used + free |
| `cpu_cached_blocks` | 当前 CPU_RESIDENT 的逻辑块；不包含过期 CPU backing 或 TRANSFERRING |
| `logical_blocks` / `transferring_blocks` | 活跃逻辑块 / 迁移或排队中的块 |
| `gpu_to_cpu_bytes` / `cpu_to_gpu_bytes` | runtime 成功完成的 D2H/H2D 字节数；包括 create 初始 H2D |
| `*_batches` | 成功完成的 batch 数；分层管理器每个 batch 一块，底层 batch 可包含 N 块 |
| `*_device_ms` | 对应方向的 CUDA event 区间之和 |
| `*_latency_us` | `device_ms * 1000 / batches`，平均每 batch 传输时间 |
| `*_bandwidth_gbps` | `bytes / device_ms / 1e6`，十进制 GB/s |
| `transfer_failed_batches` | 提交/同步失败后被清理的在途 batch；参数校验拒绝不增加此项 |
| `offload_count` / `load_count` | 成功完成的块迁移；load 包含 create 初始写入 |
| `prefetch_count` / `prefetch_skipped` | 新接纳的预取块（含排队）/ 因无可用 victim 跳过的候选次数 |
| `prefetch_hits` | 由预取准备好的块第一次被 demand 使用，且到来时已 GPU_RESIDENT |
| `prefetch_misses` | demand 到来时未 GPU_RESIDENT，包括未预取、预取过晚或正在 offload |
| `prefetch_late` | miss 中已经接纳预取，但尚未完成的部分 |
| `prefetch_unused` | 预取未消费就被搬出、释放或在错误恢复中丢弃 |
| `prefetch_outstanding` | 尚未消费的预取，可能排队、传输中或已就绪 |
| `demand_count` / `gpu_demand_hits` | 成功的 acquire_gpu 次数 / 其中到来时已 GPU_RESIDENT 的次数 |
| `request_stall_ms` | 成功 acquire_gpu 的累计 host wall time，包含 load/offload、等待和方法内 CPU 开销 |

`acquire_gpu` 是 demand 的观测入口，`read_block` 内部也调用它。`create`、显式 `load`、`offload` 和 prefetch 本身不算 demand；失败的 acquire 不增加 demand/stall。数据校验与初始化必须在基准快照之外。P95 来自基准外层逐次 acquire 计时，包含调用边界开销；它和库内累计 stall 并非同一个计时区间。

普通 GPU resident 命中不计为 prefetch hit 或 miss。关闭预取时，CPU demand 仍计 miss，而 A 全 resident 时两者都为 0。不要把 `hit/(hit+miss)` 当作全局 cache 命中率；全局 GPU 命中率是 `gpu_demand_hits/demand_count`。一个预取只在首次使用时记一次 hit 或 late，之后的 GPU 命中属于普通命中。

对完整生命周期可检查：

```text
gpu_blocks_total = gpu_blocks_used + gpu_blocks_free
demand_count = gpu_demand_hits + prefetch_misses
prefetch_count = prefetch_hits + prefetch_late + prefetch_unused + prefetch_outstanding
GPU → CPU bytes = offload_count × block_bytes
CPU → GPU bytes = load_count × block_bytes
```

最后两个等式适用于无设备错误的分层 runtime；底层 batch runtime 的计数按 batch。CUDA 发生提交或执行错误时，未确认成功的 batch 不进入成功字节/时间累计，失败数保守计入该次清理的全部在途 batch，不能据此还原已部分完成的物理 PCIe 字节数。

## 如何计时与复用 event

每个异步 batch 在 transfer stream 上用一对 CUDA events 包围其复制命令，同步成功后读取 elapsed time。完成 batch 的 event 被回收复用，池增长到历史最大并发 batch 数；不在每次常态迁移时反复创建/销毁 event。runtime 退出时销毁池。统计本身不添加逐块同步，也不改变 compute/transfer 的依赖。

Event 时间包含这段设备时间轴上的传输和可能的主机提交间隙，不含之前排队等待和 CPU staging。它不同于 request stall，也不同于整个 workload wall time。没有传输样本时 latency/bandwidth 输出 0，不表示硬件性能为零。成功计数在 `synchronize` 或 manager 的 `poll/wait` 推进后可见；即使 DMA 已在设备完成，未推进元数据的 demand 仍保守计 miss。

本版 metrics 默认开启。CPU 时钟、event record 和事件读取都有开销；所有 A/B/C 和预取模式使用相同计时实现。当前没有把这些观测成本单独剥离，也不提供服务端 metrics exporter。

## 实验矩阵

| Workload | 唯一块数 | GPU 容量 | 初始化 offloads | 稳态要观察什么 |
| --- | ---: | ---: | ---: | --- |
| A | 100 | 100 | 0 | 所有块可驻留，迁移为 0 |
| B | 100 | 80 | 20 | 有局部性时可复用，观察 offload 与预取 |
| C | 100 | 30 | 70 | 热集合放不下，观察迁移放大与传输开销 |

每块 1 MiB，CPU backing 为 100 MiB。初始 GPU 包含最后 capacity 个块，其他块在 CPU。初始化的迁移计数单独保存在 `initial_offloads`，主表为初始化后的增量。每个配置预热完整 trial 一次，正式运行 5 次，每个 trial 都重新建立相同初始状态；轮换容量及预取模式的顺序。5 次样本保留原始值、中位数与总时间 min/max，不对微小差别做显著性声明。

- 局部性轨迹：`([0..59] * 3 + [60..99]) * 3`，660 次 demand。60 块热集合在 B 中能放下，在 C 中不能。
- 循环扫描轨迹：`[0..99] * 3`，300 次 demand。明确展示 80/100 的 LRU 也可能每次 miss。
- 每条轨迹比较 next-0（无预取）、next-1、next-4，以及 compute iterations=0/500000。
- 共 3 容量 × 2 轨迹 × 3 预取 × 2 计算量 = 36 配置，180 个正式样本；访问序列也导出 CSV。

计算使用独立 scratch 的 fake compute，不读取 KV、不执行 attention。每次访问持有 GPU lease，kernel 完成后释放；所有模式都轮询，未插入 sleep。总耗时包含预取提交、query/poll 调度和尾部迁移。scheduler CPU time 只累计 prefetch/poll 函数调用时间，不等于全部 CPU busy polling 时间。最终对 100 块逐字节回读验证在计时之外。

另运行传输基准，测量 4 KiB–256 MiB 的 pageable/pinned/staged、同步与异步计算重叠、1–32 块 batch；这些指标回答基础传输优化是否有意义。A/B/C 则回答整个分层策略的代价。

## 复现

下列编译器和 SM 86 对应本机 CUDA 12.4 / RTX 3060；其他机器调整路径和架构。运行时必须可访问 GPU。

```bash
cmake -S . -B build-v1 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON -DKVFLUX_BUILD_BENCHMARKS=ON
cmake --build build-v1 -j 4
ctest --test-dir build-v1 --output-on-failure
./build-v1/kvflux_tiered_demo

# 顺序运行，避免多个 GPU 基准互相争用设备。
./build-v1/benchmark/kvflux_cache_benchmark benchmark/results/v1-local
./build-v1/benchmark/kvflux_transfer_benchmark benchmark/results/v1-local/transfer

python3 -m venv /tmp/kvflux-v1-plot
/tmp/kvflux-v1-plot/bin/python -m pip install -r scripts/requirements.txt
/tmp/kvflux-v1-plot/bin/python scripts/plot_transfer_benchmarks.py benchmark/results/v1-local/transfer
/tmp/kvflux-v1-plot/bin/python scripts/plot_cache_benchmarks.py benchmark/results/v1-local

compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 ./build-v1/tests/kvflux_async_tests
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 ./build-v1/tests/kvflux_tiered_tests
```

构建目录名称不影响产物；本次验收复用了已配置 GCC 13 的 `build-v1.8`，CMake 项目版本已是 **1.10.0**。真实结果见 [v1 最终报告](../benchmark/results/rtx3060-v1/report.md)。

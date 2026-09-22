# v1.3–v1.5：Pinned staging、异步传输与 CUDA streams

## 实现主线

```text
CPU normal buffer
  │ PinnedBuffer::copy_from（CPU 拷贝，可计入 staging 耗时）
  ▼
长期复用的 PinnedBuffer
  │ write_batch → cudaMemcpyAsync
  ▼
transfer stream ─── block 0 ─ block 1 ─ … ─ block N-1 ──┐
                                                       │ synchronize()
compute stream  ─── 独立 fake compute ──────────────────┤
                                                       ▼
                                            标记写入完成并归还运行时引用
```

`PinnedBuffer` 使用 `cudaHostAlloc`，析构时 `cudaFreeHost`，可在很多请求之间复用。`copy_from/copy_to` 提供 normal ↔ pinned staging 拷贝；如果上游能直接产生数据到 pinned buffer，可省去这一次 CPU 拷贝。

`CudaStream` 以 `cudaStreamNonBlocking` 创建具名用途的独立 stream。`AsyncTransferRuntime` 拥有 compute 和 transfer 两个 stream，通过 `cudaMemcpyAsync` 提交传输。pinned host memory 和不同的非默认 stream 是实现可重叠传输的基础，具体重叠能力仍取决于设备。参考 [NVIDIA CUDA Best Practices：异步传输与重叠](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/)。

## 最小使用方式

完整可运行示例见 [examples/async_transfer.cpp](../examples/async_transfer.cpp)。

```cpp
kvflux::GpuMemoryPool pool(32, 4096, 16);
kvflux::AsyncTransferRuntime runtime(pool); // 必须比 pool 先销毁
kvflux::PinnedBuffer input(2 * 4096), output(2 * 4096);
std::vector<unsigned char> normal(2 * 4096, 42);
std::vector<kvflux::BlockHandle> blocks{pool.allocate(), pool.allocate()};
input.copy_from(normal.data(), normal.size());
runtime.write_batch(blocks, input); // 提交 2 个 block，主机不逐块等待
// 此处可以执行 CPU 工作，或向 compute_stream 提交独立 kernel。
runtime.synchronize();
runtime.read_batch(blocks, output);
runtime.synchronize();
output.copy_to(normal.data(), normal.size());
for (auto block : blocks) pool.release(block);
```

一个 batch 对应一个连续 host buffer，大小必须恰好为 `N * block_bytes`。第 i 段对应 `blocks[i]`，GPU id 可以不连续或乱序，但同一 batch 不允许重复 id。多个不相交 batch 可以连续提交，最后只同步一次。

同一个 transfer stream 内仍按序执行 DMA。这里的“batch async”消除的是每块的 CPU 等待，不承诺 N 次同方向拷贝在硬件上同时执行。相同 PCIe 链路的带宽上限不会因为增加 stream 而消失。

## 在途数据的所有权

异步提交之后，函数返回不代表设备已经用完内存，因此 runtime 为每个块额外 retain，并持有 pinned storage 的共享引用：

- 调用方释放自己的 block 引用后，DMA 仍然持有引用，allocator 不会提前复用槽位。
- 调用方销毁 `PinnedBuffer` 对象后，底层 storage 保留到传输同步结束。
- 在途期间 `PinnedBuffer::data/copy_from/copy_to` 拒绝 host 访问；此前获得的裸指针也不得读写。
- pool 的同步 read/write/publish，以及其他 runtime 对同一块的提交会被拒绝，直到该 runtime synchronize。
- 写入在 synchronize 成功后才标记 initialized；部分提交或执行失败时不发布写入数据。
- runtime 析构会等待 compute stream 和 transfer stream，但设备错误只能通过显式 synchronize 抛出给调用方。

仍采用单 host 线程约定。pool 必须比 runtime 活得更久；本版没有把 pool 整体改成共享所有权。原有裸句柄仍要求正确成对 retain/release，不能误释放 runtime 的内部引用。

`synchronize()` 只等待 transfer stream 并更新 host 元数据；它不代表外部 kernel 已完成。原始 compute kernel 的输入/输出生命周期由调用者管理，使用完块之后应先 `runtime.compute_stream().synchronize()` 再释放引用。

`compute_wait_for_transfers()` 使用 event 建立设备侧依赖，让随后提交的计算等待已经排队的传输，不阻塞 CPU。这适合“拷贝当前输入后计算”，但有真实数据依赖的这两步不能重叠；性能基准使用独立 scratch 数据模拟计算。单独调用原始 stream synchronize 也不会更新 pool 状态，仍应调用 runtime synchronize。

## 基准设计与计时口径

源代码见 [transfer_benchmark.cpp](../benchmark/transfer_benchmark.cpp)。所有显存和 pinned buffer 在计时之前分配；每种模式预热两次，轮换模式测试顺序，记录 9 个样本的中位数。不把分配成本混进 steady-state 传输指标。

### v1.3：normal vs pinned

大小为 **4 KiB、64 KiB、1 MiB、16 MiB、256 MiB**，同时测量 H2D 与 D2H。

| 模式 | 计时包含的操作 |
| --- | --- |
| pageable | normal host ↔ GPU，同步完成 |
| pinned_direct | 已准备好的 pinned host ↔ GPU，同步完成 |
| pinned_staged | H2D 包含 normal→pinned→GPU；D2H 包含 GPU→pinned→normal |

latency 为 host wall clock 微秒，包含 API、验证和等待。小尺寸每个样本循环 100 次，1/16 MiB 循环 5 次，256 MiB 循环 1 次，除以循环次数得到单次延迟。`bandwidth_GB/s = bytes / latency_us / 1000`，GB/s 使用十进制。

CPU staging 不是免费操作。不能拿 pinned_direct 的成绩声称 normal→pinned→GPU 的端到端路径一定更快。

### v1.4：sync vs async

固定 16 MiB pinned H2D，计算使用独立 device scratch 上的浮点递推 kernel，每线程 500000 次迭代，128 threads/block，grid 为 SM 数的两倍。结果写回并验证，避免无效计算被编译器删掉。

- sync：传输完成后才提交 compute。
- async：两个 non-blocking streams 独立提交，最后等待两者完成。
- total runtime：从提交到两条 stream 均完成的 host wall time。
- transfer time / compute time：CUDA events 测量各 stream 的区间。
- effective bandwidth：传输字节数 / **包含计算的总时间**。
- transfer bandwidth：传输字节数 / transfer event 时间。
- overlap ratio：两个 event 区间交集 / 较短区间长度。0 表示无区间重叠，1 表示较短区间完全被覆盖。

跨 stream 事件共享一个起始 event 时间轴。event 区间可能包含主机提交间隙，overlap ratio 是区间重叠指标，不是硬件利用率，也不证明任意模型计算能同样重叠；精确的 engine 活跃情况需进一步用 profiler 检查。

### v1.5：batch throughput

每块 1 MiB，测试 **1、2、4、8、16、32** 块。sync_per_block 使用 pinned buffer 逐块复制并等待；async_batch 用 runtime 一次提交 N 个块，最后等待一次。记录总时间和有效 GB/s，覆盖提交/引用管理的 host 开销。测试后读回全部块检查字节一致性。

## 构建与复现

核心库、异步测试与示例只使用 CUDA Runtime，不依赖自定义 kernel：

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON \
  -DKVFLUX_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
./build-cuda/kvflux_async_demo
```

完整性能基准包含 fake compute kernel，需要 CUDA Toolkit 支持的 host compiler。下面对应本机 CUDA 12.4、GCC 13 和 RTX 3060（SM 86）；其他机器替换编译器路径和架构值：

```bash
cmake -S . -B build-cuda-release -DKVFLUX_ENABLE_CUDA=ON \
  -DKVFLUX_BUILD_TESTS=ON -DKVFLUX_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-release -j 2
./build-cuda-release/benchmark/kvflux_transfer_benchmark benchmark/results/local

python3 -m venv /tmp/kvflux-plot-venv
/tmp/kvflux-plot-venv/bin/python -m pip install -r scripts/requirements.txt
/tmp/kvflux-plot-venv/bin/python scripts/plot_transfer_benchmarks.py benchmark/results/local
```

输出 `transfers.csv`、`overlap.csv`、`batch.csv`、环境信息、Markdown 报告及 PNG/SVG 图。运行时需能访问 NVIDIA 驱动。本次保存的真实结果见 [RTX 3060 报告](../benchmark/results/rtx3060/report.md)。

## 本机结果说明

RTX 3060 Laptop GPU、CUDA Runtime 12.4，本次 256 MiB H2D：pageable 5.52 GB/s，pinned_direct 6.73 GB/s，pinned_staged 4.60 GB/s。4 KiB H2D 的 pinned 反而略慢。结论是直接使用长期 pinned buffer 有收益，临时 staging 的额外拷贝可能抵消收益。

16 MiB + fake compute：sync 3.61 ms，async 2.41 ms，约 1.49× 加速；event 区间 overlap ratio 为 100%，表示较短的计算区间被传输区间覆盖。

32 块 H2D：同步 6.55 GB/s，异步 batch 6.80 GB/s。提升不大，2 块时本次测量甚至变慢；这说明减少 host 等待不等于按 batch 数线性提速。当前只有一次基准运行的多次样本中位数，没有置信区间，不应把微小差别当成跨环境保证。

## 验证与后续工作

真实 GPU 测试覆盖 1/2/4/8/16/32 块、乱序 id、双向逐字节一致性、在途 host 访问限制、无效 batch 原子拒绝、提前销毁 pinned 对象、提前 release 块、runtime 析构收尾、跨 stream event 依赖和缓存写保护。Compute Sanitizer 检查异步生命周期路径无越界、无泄漏。

下一步可实现 bounded pinned buffer pool、staging ring/double buffering、连续块合并传输、完成事件或 future 级别的回收，以及将原始 block 句柄改为 RAII 所有权。当前每个 batch 要占用一个不在途的 buffer；不提供并发 host 调用、请求取消或真实 transformer 接入。

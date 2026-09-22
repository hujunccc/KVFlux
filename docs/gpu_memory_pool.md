# v1.0–v1.2：真实 GPU KV Cache Pool

## v1.0：BlockId 对应一段长期存在的显存

`GpuMemoryPool` 构造时使用一次 `cudaMalloc` 申请整个池，析构时使用 `cudaFree` 释放。每次请求的 allocate/release 只操作内部 v0 `BlockManager` 的槽位、引用和 LRU，不重新申请设备内存。

```text
GpuMemoryPool
├── BlockManager：负责哪些 BlockId 可用
└── 一次申请的连续 GPU memory
    ├── block 0 : base + 0 × block_bytes
    ├── block 1 : base + 1 × block_bytes
    └── block 17: base + 17 × block_bytes
```

```cpp
kvflux::GpuMemoryPool pool(128, 4096, 16); // 128 块，每块 4096 bytes、16 tokens，device 0
const auto h = pool.allocate();
void* gpu = pool.device_address(h); // 验证 generation 和活跃引用
const auto total = pool.total_blocks();
const auto remaining = pool.remaining_blocks();
pool.release(h);
```

`total_blocks()` 与地址查询为 O(1)。`remaining_blocks()` 返回 `free + cached_idle`，即可以立即分配或淘汰后分配的数量。若需要区分这两种块，使用 `stats()`；当前统计仍是 O(capacity) 扫描。

也提供 `device_address(BlockId)` 查询原始映射，只检查 id 范围。GPU 指针不能在 CPU 上解引用；原始指针不代表持有引用，也不能在池销毁后使用。面向请求应优先传 `BlockHandle`，这样能拒绝槽位复用后的旧句柄。

当前是单线程、单池单设备设计。设备操作会临时切换到池所在设备并恢复调用者设备。不调用 `cudaDeviceReset`，池应在外部 reset 之前销毁。析构不抛异常；CUDA 上下文被提前销毁或设备故障时，无法保证清理成功。

## v1.1：统一布局与容量公式

统一紧密布局：`[2][num_layers][num_heads][block_size][head_dim]`。

第一维的 0 是 K，1 是 V。因此先连续存所有层的 K，再连续存所有层的 V。head_dim 是最快变化的维度，无 padding。`num_heads` 特指 KV heads；GQA 模型不能误填 query heads。

```text
element_bytes = FP16: 2 / BF16: 2 / FP32: 4
block_bytes = 2 × layers × kv_heads × block_size × head_dim × element_bytes
total_blocks = floor(budget_bytes / block_bytes)
allocated_bytes = total_blocks × block_bytes
token_capacity = total_blocks × block_size
```

`KVBlockLayout::byte_offset(kind, layer, head, token, dimension)` 返回块内字节偏移，V 的起点是 `value_offset()`。构造时检查零维度、未知 dtype 和乘法溢出，查询时检查每个下标。

```cpp
kvflux::KVBlockLayout layout({32, 8, 128, 16, kvflux::DType::Float16});
auto plan = layout.capacity_for(std::size_t{1} << 30); // 1 GiB
// block_bytes=2097152, total_blocks=512, token_capacity=8192
kvflux::GpuMemoryPool pool(layout, std::size_t{1} << 30);
```

估算不需要 GPU。`capacity_for()` 对不足一块的预算返回 0 块；真正构造 pool 会拒绝零容量。用户预算不是当前空闲显存的保证，实际 `cudaMalloc` 失败时抛出包含 CUDA 错误信息的异常，不静默缩小池。

示例使用 GiB（2^30 bytes），不是 GB（10^9 bytes）。token_capacity 是总物理 token 槽位，不是模型最大上下文，也不包含模型权重、工作空间和其他 GPU 开销。

## v1.2：完整块同步读写

```cpp
kvflux::KVBlockLayout layout({2, 4, 8, 16, kvflux::DType::Float32});
kvflux::GpuMemoryPool pool(layout, 20 * layout.block_bytes());
auto h = pool.allocate();
std::vector<float> a(layout.block_bytes() / sizeof(float), 1.25f), b(a.size());
pool.write_block(h, a.data(), layout.block_bytes());
pool.read_block(h, b.data(), layout.block_bytes());
// a == b
pool.release(h);
```

读写以字节为单位，长度必须等于整块长度，host 指针不能为 null；调用者必须保证缓冲区至少有指定长度。FP16/BF16/FP32 只决定元素宽度，不做数值或 dtype 转换。测试对原始字节精确比较，因此保留所有位模式。

H2D 使用 `cudaMemcpy` 后显式等待默认 stream，D2H 使用同步复制，函数返回后可以安全重用 host 缓冲区。本版不提供异步 stream API，不能一边外部 kernel 访问同一块一边调用 release/读写。CUDA 语义参考 [NVIDIA Runtime Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)。

为了避免读到旧数据或破坏其他请求：

- 新分配或淘汰复用的块标记为未初始化，read/publish 会拒绝它。
- 完整 write 成功后才标记为已初始化；复制失败则保持未初始化。
- write 要求 refs=1 且未发布，禁止覆盖共享块或前缀缓存。
- prefix publish 必须发生在 write 成功后。lookup 命中保留数据并增加引用。
- release 后不清零显存，复用时重置初始化状态，下一次必须整块覆盖。

GPU pool 不暴露 v0 的模拟 `acquire(tokens)`；该方法会提前发布元数据，不适合真实 K/V。正确路径为 `lookup → 未命中时 allocate → write_block → publish → release`。直接拿原始 GPU 指针写入会绕过保护，目前也没有用于外部 kernel 的完成通知接口。

## 构建与验收

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON \
  -DKVFLUX_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
./build-cuda/kvflux_gpu_demo
./build-cuda/kvflux_capacity 1
```

只使用 CUDA Runtime API，没有自定义 kernel，因此 C++ 编译器链接 `CUDA::cudart` 即可，无须开启 CUDA 编译语言。显式开启 CUDA 时缺少 Toolkit 会配置失败，不会偷偷使用 host mock。未开启 CUDA 时仍可构建 v0 和布局/容量估算。

测试覆盖三种 dtype、单块、20 块（含 id 17）、8 轮连续覆盖写与随机读取、地址间距、真实 device memory 属性、池耗尽、释放后复用、generation、防未初始化访问、共享与发布后写保护、缓存命中和 LRU 淘汰。没有 CUDA device 时 GPU 用例显示 skip；驱动初始化出错则失败。多卡恢复检查仅在设备数超过 1 时执行。

本轮验收环境：2026-09-22，GCC 15.2、CUDA Toolkit 12.4、NVIDIA RTX 3060 GPU。CUDA Debug 下 3/3 CTest 通过；CPU Release 下 2/2 通过；CPU AddressSanitizer/UndefinedBehaviorSanitizer 下 2/2 通过。NVIDIA Compute Sanitizer 的 memcheck + leak-check 报告 0 errors、0 bytes leaked。单卡环境未执行多卡分支。

复现设备内存检查：

```bash
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 \
  ./build-cuda/tests/kvflux_gpu_tests
```

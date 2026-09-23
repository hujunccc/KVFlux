# v2 Milestone 6：Paged KV Storage

`PagedKVStorage` 把 v2 的物理页编号接到 v1 `GpuMemoryPool` 的真实显存分配。第一版采用**单层**紧密布局；一个实例保存一层的 K 和 V：

```text
K[num_blocks][num_kv_heads][block_size][head_size]
V[num_blocks][num_kv_heads][block_size][head_size]
```

一次 v1 显存分配中，先放全部 K 页，再放全部 V 页。`head_size` 最快变化，没有 padding。对 `physical_block=p`、KV head `h`、页内 token `t`、维度 `d`：

```text
element_bytes = FP16/BF16: 2；FP32: 4
page_bytes    = num_kv_heads × block_size × head_size × element_bytes
cache_bytes   = num_blocks × page_bytes
K byte offset = (((p × num_kv_heads + h) × block_size + t) × head_size + d) × element_bytes
V byte offset = cache_bytes + K byte offset
```

`PagedKVLayout` 只算形状和偏移，CPU 上可运行；它检查维度、dtype、越界坐标和乘法溢出。布局与 v1 的块内 `[K/V][layer][head][token][dimension]` 不同：这里 K 与 V 各有完整的 `[block][head][token][dimension]` 区域。多层模型可为每层创建一个存储实例，使用同一个 v2 物理页池和编号。

## 连接页池与 Slot Mapping

`PhysicalBlockPool` 是**唯一的页编号与引用所有者**。`PagedKVStorage` 内部复用 v1 `GpuMemoryPool` 做一次性显存分配，只调用其按编号查询原始地址的接口，不调用它的独立分配器。v2 页 `p` 对应 K 区第 `p` 页、V 区第 `p` 页。

`page_address(kind, handle)` 和 `element_address(kind, handle, ...)` 先经 v2 页池校验句柄及 generation。Milestone 5 的 `slot_mapping` 给出 `slot = p × block_size + t`；`slot_address(kind, slot, head, dimension)` 把它转换成 K/V 设备地址。slot 只携带编号，不携带引用，因此调用方必须在 GPU 读写完成前保持请求的页引用存活。返回的设备指针不能在 CPU 上解引用；地址查询也不表示内容已初始化。

当前存储层已经可以通过这些设备地址真实写入、读回 K/V；测试用 CUDA 拷贝验证每个布局维度。[Paged KV Write](v2_paged_kv_write.md) 已加入设备端写入 kernel 和 slot mapping 上传。异步完成事件、设备读取和共享页写保护仍由后续计算/调度层完成。布局选择以正确性和易读性为先；更复杂的 coalescing 布局可在 kernel 优化阶段再做。

## 构建与验证

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure

cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
./build-cuda/kvflux_paged_kv_storage_demo
```

无 CUDA 时可测试 `PagedKVLayout` 公式；真实显存测试需要可访问的 NVIDIA GPU。

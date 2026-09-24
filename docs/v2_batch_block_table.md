# v2 Milestone 11：Batch Block Table

单请求的 Block Table 只有一个维度。batch decode 需要让 GPU 同时知道请求下标、
该请求的逻辑块下标和有效 token 数。`build_batch_block_table` 按传入请求顺序
生成行主序二维表，以及同顺序的 `sequence_lengths`：

```text
block_table[request][logical_block]
    = flat_table[request * max_blocks_per_sequence + logical_block]

请求       L0   L1   L2   L3      sequence_lengths
A         12   87   32   -       35
B         76    3   45  18       61
C         90   41    -   -       28
```

短行用 `BatchBlockTable::invalid_block()` 填充。GPU 根据
`sequence_lengths[request]` 计算本请求最后一个有效 token，只读取对应的有效
逻辑块，不把 padding 当成物理页。CPU 的 `physical_id(request, logical)`
也会拒绝越界或 padding。空请求可出现在元数据快照中；空 batch 的表宽为 0。
构造时拒绝空指针和来自不同物理页池的请求。

`paged_attention_batch` 消费这两段元数据，当前提供每条非空请求一个末尾 Q
的 batch decode。Q/O 形状为 `[num_sequences][query_heads][head_size]`，
顺序与请求列表一致。kernel 对每个输出元素从 `sequence_lengths[request]-1`
向前遍历 KV，按 `request * max_blocks_per_sequence + token / block_size`
查物理页，再读取该页的 K/V。每次调用上传表和长度，在默认 stream 执行并
同步返回。Q/O 必须是存储所在 GPU 的独立设备缓冲，且不能与缓存重叠；
请求、页和映射在调用期间必须保持有效。空 batch 不访问 Q/O。

```cpp
std::vector<const kvflux::v2::SequenceState*> batch{&request_a, &request_b};
auto table = kvflux::v2::build_batch_block_table(batch);
// table.block_table() 是行主序的物理页编号；table.sequence_lengths() 是 token 数。
kvflux::v2::paged_attention_batch(storage, batch, query_device, output_device,
                                  query_heads);
```

CPU 测试使用长度 `[35, 61, 28, 0]` 和交错分配的非连续物理页，检查行宽、
padding、下标错误及不同页池。CUDA 测试用三条长度 `[5, 9, 2]` 的请求，
检查 FP32、FP16、BF16 的 batch decode 输出与逐请求 CPU reference attention
一致，并检查未用尾页、输出边界、非法参数。当前 API 聚焦单步 batch decode；
每请求不同数量的 Q 与调度器集成留给后续阶段。

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure \
  -R 'kvflux_(batch_block_table|paged_attention_batch)_tests'
```

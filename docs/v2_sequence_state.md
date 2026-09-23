# v2 Milestone 3：SequenceState

`SequenceState` 为每个请求建立独立的逻辑 KV 地址空间。请求记录 `request_id`、`num_tokens`，并拥有一张 [Block Table](v2_block_table.md)；表项持有全局 [Physical Block Pool](v2_physical_block_pool.md) 中物理页的引用。请求销毁时，表会归还这些引用。

以 `block_size=16` 的请求 `1001` 为例，追加 35 个 token 后：

```text
request_id             1001
num_tokens             35
num_allocated_blocks    3
last_block_num_tokens   3
logical block           0    1    2
physical block          7   33   91
```

实际物理编号由池的分配状态决定，不要求连续。token 下标 `i` 的位置为：

```text
logical_block = i / block_size
offset        = i % block_size
physical_page = block_table[logical_block]
```

例如 token 34 位于逻辑块 2、块内偏移 2，可映射到物理页 91。`token_location(i)` 会返回物理页编号和块内偏移；`i >= num_tokens` 时拒绝访问。

## 状态与追加

请求从空表和零 token 开始。`append_tokens(count)` 先按需申请物理页，成功后才更新 `num_tokens`。若申请或表扩容失败，本次新页会被归还，原有 token 数、映射和引用保持不变。尾块还有空间时，追加只增加 token 数；满块的 `last_block_num_tokens` 返回 `block_size`，空请求返回 0。

`num_allocated_blocks` 直接取表长，始终等于 `ceil(num_tokens / block_size)`；`last_block_num_tokens` 由 token 数计算，因此不会与表产生两份独立状态。表只对外提供只读访问。请求可以移动，不可复制；物理页池必须比请求活得更久。

单 token decode 追加与非连续物理页的例子见 [Milestone 4：Dynamic Sequence Growth](v2_dynamic_sequence_growth.md)。请求本身只管理逻辑 token 位置和物理页编号；[Paged KV Storage](v2_paged_kv_storage.md) 负责真实 GPU 地址，[Paged KV Write](v2_paged_kv_write.md) 负责设备端写入。共享尾块的写时复制和计算调度属于后续阶段。

## 运行

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./build/kvflux_sequence_state_demo
```

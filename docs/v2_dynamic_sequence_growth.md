# v2 Milestone 4：Dynamic Sequence Growth

decode 时，请求每次新增一个 token。`SequenceState::append_token()` 为它预留位置，并返回 `TokenLocation {physical_block, offset_in_block}`。已有尾块未满时不申请物理页；只有跨过块边界才向全局 `PhysicalBlockPool` 申请一页并追加到 Block Table。

`block_size=16` 时，请求 A 的状态变化如下：

| token 数 | Block Table | 新 token 的位置 | 新申请物理页 |
| ---: | --- | --- | --- |
| 15 | `[P37]` | — | — |
| 16 | `[P37]` | `P37`，偏移 15 | 否 |
| 17 | `[P37, P91]` | `P91`，偏移 0 | 是 |

逻辑块 0 和 1 始终按请求内顺序排列，物理页 37 和 91 可以相距很远。追加过程只操作 Block Table 和页池元数据，无需搬动 P37 中已有 token。`num_allocated_blocks` 从 1 变为 2，尾块占用数从 16 变为 1。

单 token 追加沿用 [SequenceState](v2_sequence_state.md) 的分配和回滚规则：池已满时抛 `CapacityError`，token 数、Block Table 和已有物理页引用保持原样。`append_tokens(count)` 仍可用于一次预留多个 token，例如 prompt 初始化；调用失败时本次新申请的页会全部归还。

返回的位置标识物理页及页内 token 偏移；[Paged KV Storage](v2_paged_kv_storage.md) 将它换算成真实 GPU 地址，[Paged KV Write](v2_paged_kv_write.md) 负责设备端写入。共享尾块的追加现在由 [Copy-on-Write](v2_copy_on_write.md) 路径处理。

## 运行示例

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./build/kvflux_dynamic_growth_demo
```

示例用其他页的占用制造非连续编号，并打印 `15 → 16 → 17` 的表变化。请求及占用者释放后，100 个物理页全部恢复可用。

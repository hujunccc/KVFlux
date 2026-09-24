# V2 正确性测试矩阵

以下名称对应设计要求，右列是实际由 CTest 注册的测试。GPU 测试覆盖 FP16、BF16、FP32；没有可用 CUDA 设备时返回 77，CTest 显示 `Skipped`，不能视为 GPU 路径已通过。

| 要求 | CTest 名称 | 核心断言 |
| --- | --- | --- |
| BlockTableTest | `kvflux_block_table_tests` | 逻辑块索引找到对应物理块；越界和引用所有权正确。 |
| AppendTest | `kvflux_sequence_state_tests` | 满块前复用尾页，跨块时新增页，失败时回滚。 |
| NonContiguousTest | `kvflux_fragmentation_tests`、`kvflux_paged_kv_read_tests` | 非连续物理编号仍可建立逻辑页表并读取 K/V。 |
| ReuseTest | `kvflux_physical_pool_tests` | 释放后可重新分配，旧 generation 句柄失效。 |
| IsolationTest | `kvflux_paged_kv_gather_tests`、`kvflux_copy_on_write_gpu_tests` | 交错写入的两个请求独立读回；写时复制后两侧新 token 不互相覆盖。 |
| SlotMappingTest | `kvflux_slot_mapping_tests` | `(request, token)` 映射到正确物理 slot，保持 batch 输入顺序。 |
| PagedWriteTest | `kvflux_paged_kv_write_tests` | CUDA 写入 K/V 后逐元素回读，邻近 slot 不被覆盖。 |
| PagedReadTest | `kvflux_paged_kv_read_tests` | `[27, 3, 91]` 页表按逻辑顺序读取，只返回尾页有效 token。 |
| PagedGatherTest | `kvflux_paged_kv_gather_tests` | 交错 batch 写入后，分别 gather A/B，与独立构造的连续 K/V 精确一致。 |
| PagedAttentionTest | `kvflux_paged_attention_tests`、`kvflux_paged_attention_batch_tests` | 非连续页上的输出逐元素对比 CPU 连续 K/V reference；覆盖 prefill、decode、batch。 |
| BatchTest | `kvflux_batch_block_table_tests` | 多请求二维页表、padding 与 `sequence_lengths` 正确。 |
| RefCountTest | `kvflux_prefix_cache_tests`、`kvflux_physical_pool_tests` | 共享前缀引用计数、最后一次释放、缓存页复用与淘汰正确。 |
| COWTest | `kvflux_copy_on_write_tests`、`kvflux_copy_on_write_gpu_tests` | 共享 partial 尾页只复制写入方；分配/复制失败回滚；真实 GPU K/V 双向隔离。 |
| FragmentationTest | `kvflux_fragmentation_tests` | 128 页池中 64 个互不相邻的空闲页都能分配给同一请求，释放后全部回收。 |
| OOMTest | `kvflux_oom_tests` | 容量不足抛 `CapacityError`，部分分配回滚，已有映射与引用计数保持不变。 |

## 运行

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON -DKVFLUX_ENABLE_CUDA=OFF
cmake --build build -j 4
ctest --test-dir build --output-on-failure

cmake -S . -B build-cuda -DKVFLUX_BUILD_TESTS=ON -DKVFLUX_ENABLE_CUDA=ON
cmake --build build-cuda -j 4
ctest --test-dir build-cuda --output-on-failure
```

重点复核可以运行 `ctest --test-dir build-cuda --output-on-failure -R 'fragmentation|paged_attention|copy_on_write'`。如 GPU 用例显示 `Skipped`，需要在有可用 CUDA 设备的机器上重跑。

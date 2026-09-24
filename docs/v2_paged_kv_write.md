# v2 Milestone 7：Paged KV Write

这是 v2 第一条 CUDA kernel 数据写路径。控制面先完成请求增长和 [Slot Mapping](v2_slot_mapping.md)，然后 `write_paged_kv(storage, K_new, V_new, slot_mapping)` 将本 batch 的新 K/V 写入 [Paged KV Storage](v2_paged_kv_storage.md)。

```text
请求 token 位置 → Block Table → 物理页 → slot_mapping
                                     ↓
设备端 K_new / V_new → CUDA kernel → GPU K Cache / V Cache
```

## 输入与地址

- `K_new`、`V_new` 是**同一 GPU 上的设备指针**，形状均为 `[batch][num_kv_heads][head_size]`，dtype 与存储布局一致。
- `slot_mapping` 是控制面生成的 `std::vector<uint64_t>`，长度等于 batch；第 `i` 项指定输入第 `i` 个 token 的目标槽位。
- 存储仍采用单层 `K/V[num_blocks][num_kv_heads][block_size][head_size]`。kernel 对每个 batch token、KV head 和维度，按 `block = slot / block_size`、`token = slot % block_size` 写入 K 和 V 的相同物理槽位。

例如控制面生成 `[173, 522, 944]`，三组输入分别落在 P10 的 token 13、P32 的 token 10、P59 的 token 0；物理页无需连续。FP16/BF16 的 16 位原始值和 FP32 的 32 位原始值按位写入，不做数值转换。

第一版每次调用上传 slot 数组，启动 kernel，等待默认 stream 完成后返回。输入需已在默认 stream 上就绪；若由其他 stream 产生，调用方先同步。空 batch 不访问输入指针；空指针、越界或重复 slot、非目标 GPU 设备指针会被拒绝。调用方需保证输入缓冲长度足够，且请求页在调用完成前保持引用；共享页的写时复制和异步调度尚未接入。返回后可安全复用输入缓冲或释放请求页。

## 运行

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
./build-cuda/kvflux_paged_kv_write_demo
```

真实 GPU 测试从三个 `SequenceState` 生成上述 slot，检查所有 head/dimension 的 K/V 逐字节读回，以及相邻槽位不被覆盖。后续 [Paged KV Read](v2_paged_kv_read.md) 已提供按 Block Table 逐页读取的 GPU 路径，[Copy-on-Write](v2_copy_on_write.md) 已处理共享尾块的追加；跨层批量调度和异步完成事件仍属后续阶段。

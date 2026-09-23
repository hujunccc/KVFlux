# v2 Milestone 5：Slot Mapping

控制面先让请求预留待写 token 的位置，再把 decode batch 转成一段连续的 `slot_mapping`。GPU 侧将来只需按 batch 下标读取槽位编号，无须遍历 C++ `SequenceState` 和 `BlockTable`。

```text
SequenceState + token_position
    → logical_block = token_position / block_size
    → physical_block = block_table[logical_block]
    → slot = physical_block * block_size + token_position % block_size
```

`slot` 是**token 槽位编号**，不是 CUDA 指针或字节偏移。实际 K/V 地址由 [Paged KV Storage](v2_paged_kv_storage.md) 的页布局计算。

## Batch 接口

`build_slot_mapping(batch)` 接收按写入顺序排列的 `{sequence, token_position}`，返回等长的 `std::vector<std::uint64_t>`。输出下标与输入 batch 下标一一对应，可作为后续设备缓冲区的数据来源。

调用前必须用 `append_token()` 或 `append_tokens(count)` 为各请求预留位置。构建器只读请求状态，不分配物理页、不增加引用；调用期间请求须存活。GPU 计算期间保持页引用和同步的责任属于后续调度层。空 batch 返回空数组；空请求指针、越界位置、混用不同物理页池以及槽位算术溢出会被拒绝。

例如 `block_size=16`，batch 顺序为 A、B、C：

| 请求 | token 位置 | 对应物理页 | 页内偏移 | slot |
| --- | ---: | ---: | ---: | ---: |
| A | 34 | 81 | 2 | 1298 |
| B | 17 | 32 | 1 | 513 |
| C | 80 | 138 | 0 | 2208 |

这里 C 的 `slot` 是 **2208**：`138 × 16 + 0`。数值 2216 是 P138 的偏移 8，对应 token 88。slot mapping 的数值必须由实际物理页和块内偏移计算，不能单独指定。

## 运行

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./build/kvflux_slot_mapping_demo
```

当前模块在 CPU 控制面生成数组；[Paged KV Storage](v2_paged_kv_storage.md) 已能将 slot 换算为真实 GPU 地址。设备上传、K/V 写入 kernel 和执行期页生命周期管理尚未接入。

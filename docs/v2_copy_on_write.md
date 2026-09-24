# v2 Milestone 13：Copy-on-Write

`SequenceState::fork_shared(new_request_id)` 创建一个共享当前 Block Table 的新请求。
它为每个物理页增加一次引用，**包括未满的尾块**，不复制 K/V。两条请求读取
相同的历史 token 时，仍映射到同一物理页：

```text
复制前：A L2 ─┐
             ├── P71 (refcount=2，尾块未满)
        B L2 ─┘

A 追加：分配 P83 → 复制 P71 的 K/V 到 P83 → A L2 改为 P83 → 写 A 的新 token

复制后：A L2 → P83 (refcount=1)
        B L2 → P71 (refcount=1)
```

`append_token_cow(storage, a)` 是设备端入口。如果尾块未满且引用数大于 1，
它先申请新物理页，用 `PagedKVStorage::copy_block` 复制整页 K 和 V，等默认
stream 完成，再替换 A 的尾部表项并归还 A 对旧页的引用。返回的新 token
位置尚未写入 K/V，调用方随后用 `build_slot_mapping` 和 `write_paged_kv`
写入。复制失败或页池容量不足时，A/B 的 token 数、表项和引用计数保持不变。

完整共享页追加时会直接申请新尾页；未共享尾页直接原地追加，两种情况
都无需复制。普通 `append_token` / `append_tokens` 在共享 partial 尾块上
会抛 `std::logic_error`，防止调用方绕过写时复制后覆盖另一个请求的数据。

```cpp
kvflux::v2::SequenceState b = a.fork_shared(2);
// 源页 K/V 必须已写入，且与其他 stream 的工作建立完成依赖。
auto location = kvflux::v2::append_token_cow(storage, a);
// 用 a 的新 token 位置生成 slot mapping，再写入本次新 K/V。
```

CPU 元数据接口 `append_token_with_copy(callback)` 接受页复制回调；它用于
测试或其他存储后端。CUDA 入口将回调连接到真实 GPU K/V 页复制。页池与
存储必须相同；请求和物理页在 GPU 操作完成前须保持存活。当前复制完整页
并同步返回，适合验证正确性，后续可再优化复制范围和异步生命周期。

这与操作系统虚拟内存的关系相同：`SequenceState` 是请求的虚拟地址空间，
Block Table 是页表，物理 KV block 是物理页，页引用计数决定是否需要
Copy-on-Write。PrefixCache 的哈希命中仍只共享已发布的完整块；partial
共享由 `fork_shared` 显式创建。

CPU 测试检查共享尾页、分叉后隔离、完整页追加不复制、池满和复制异常
的回滚。CUDA 测试实际写入三枚 token，fork 后 A 写入第四枚，验证 B
仍读到原页；再由 B 写自己的第四枚，验证 A 的数据不变。覆盖 FP16、
BF16、FP32 的原始位模式。没有可用 NVIDIA GPU 时 CUDA 测试跳过。

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure \
  -R 'kvflux_copy_on_write(_gpu)?_tests'
```

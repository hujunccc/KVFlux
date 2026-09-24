# v2 Milestone 12：Block Sharing

`PrefixCache` 把 v0 `BlockManager` 的完整前缀哈希、引用计数、缓存索引和
空闲缓存队列接到 v2 的 `SequenceState` / `PhysicalBlockPool`。相同的完整
token 前缀可以让两个请求的逻辑块指向同一物理页：

```text
请求 A: L0 → P7,  L1 → P19, L2 → 私有尾块
请求 B: L0 → P7,  L1 → P19, L2 → 另一私有尾块
             P7 refcount=2, P19 refcount=2
```

哈希只是候选索引。命中前会比较**从请求开头到当前块末尾的全部 token**，
所以哈希碰撞、相同局部块但不同上文都不会误共享。只有完整块能发布和命中；
不足一块的尾部保持私有。请求结束时，`BlockTable` 析构归还它持有的引用。
最后一个引用归零后，已发布页保留在缓存队列中供后续请求复用；当普通
free list 耗尽时，池才淘汰最早闲置的缓存页并复用其物理编号。未发布的
私有页则直接回到 free list。

## 使用顺序

```cpp
kvflux::PhysicalBlockPool pool(128, 16);
kvflux::v2::PrefixCache cache(pool);
kvflux::v2::SequenceState a(1, pool);
a.append_tokens(prompt_a.size());
// 把 A 的有效 K/V 写入 PagedKVStorage，并等待写入完成。
cache.publish_computed_block(a, 0, first_16_tokens);
cache.publish_computed_block(a, 1, first_32_tokens);

kvflux::v2::SequenceState b(2, pool);
const auto reused = cache.attach_cached_prefix(b, prompt_b);
b.append_tokens(prompt_b.size() - reused);
// 只计算并写入 B 从 reused 开始的新 token；之后按完成情况发布新完整块。
```

`attach_cached_prefix` 只接受空请求，从第一个完整块开始连续查找，遇到
首个未命中就停止，返回已复用的 token 数。它只改动请求的 Block Table 和
引用计数，不复制 K/V；失败时回滚本次附加的引用。`publish_computed_block`
要求该块在请求中完整存在，且调用方已经把正确的 K/V 写完并同步。
如果使用其他 CUDA stream，必须先建立完成依赖，才能发布；GPU 仍在读页时
也不能释放它的最后一个引用。

一个池对应同一模型及推理配置的 KV 命名空间。调用方负责提供该请求真实的
token 前缀，且只在相同模型、适配器及位置编码配置下共享页；当前接口没有
自动校验这些外部身份，也不负责调度。共享页是只读的，`write_paged_kv`
不能对它们写入。`PrefixCache` 只共享完整块；[Milestone 13](v2_copy_on_write.md)
另外允许显式 fork 共享 partial 尾块，并在追加前执行写时复制。

该生命周期与 [vLLM 的 prefix caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)
中的 block ID、hash、ref count、block pool 和 free queue 对应。这里沿用本
项目 v0 的实现，并在写入完成后显式发布；不依赖 vLLM 源码。

## 验证

CPU 测试检查两请求共享、分叉私有尾块、引用计数、请求释放后再次命中、
强制哈希碰撞、不同完整前缀、零引用缓存页淘汰与 generation 更新。
CUDA 测试在真实 `PagedKVStorage` 写入 A 后发布，B 不再次写入而从共享页读出
相同 K/V。无 NVIDIA GPU 时该测试会跳过。

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure \
  -R 'kvflux_(prefix_cache|paged_kv_read)_tests'
```

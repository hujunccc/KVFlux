# v2 Milestone 14：Sequence 生命周期

`SequenceLifecycle` 把 v2 的物理页分配、Block Table、Paged KV 写入、
PagedAttention 和请求结束时的引用释放串成一个同步的单请求流程。调用方提供
GPU 上的 Q/K/V 输入和 O 输出；本类管理请求的 `SequenceState`：

```text
创建请求（空 SequenceState）
  → prefill：按 prompt token 数申请页 → 生成 slot mapping → 写全部 K/V
             → 因果 attention
  → decode：追加一枚 token；尾页未满则复用，满页则分配新页
            → 如尾页共享，先 Copy-on-Write → 写新 K/V → 末尾 Q attention
  → finish：释放该请求 Block Table 持有的全部物理页引用
```

例如 `block_size=2`，三枚 prompt token 占两页，第一次 decode 填满第二页，
第二次 decode 才申请第三页。每次写入前都按当前 Block Table 生成 slot；
因此物理编号不要求连续。GPU 写入和 attention 接口均同步返回，`finish()`
之后可安全归还本请求独占的页。被其他请求共享或已发布到前缀缓存的页，
仍按引用计数及缓存队列规则保留。

## 接口与输入

```cpp
kvflux::v2::SequenceLifecycle request(request_id, pool, storage, query_heads);
request.prefill(prompt_length, prompt_k_device, prompt_v_device,
                prompt_q_device, prompt_o_device);
auto location = request.decode(next_k_device, next_v_device,
                               next_q_device, next_o_device);
request.finish();
```

prefill 的 K/V 形状是 `[prompt_length][num_kv_heads][head_size]`，Q/O 是
`[prompt_length][query_heads][head_size]`。decode 各输入/输出只包含当前
一枚 token，分别是 `[num_kv_heads][head_size]` 和
`[query_heads][head_size]`。K/V dtype 与 storage 一致，Q/O 为 FP32。
这些缓冲都须位于 storage 所在 GPU；`query_heads` 必须是 KV 头数的整数倍。
输入须在默认 stream 上就绪，或由调用方先建立跨 stream 依赖。

`prefill` 只能执行一次，并要求至少一枚 prompt token。`decode` 只能在
prefill 成功后调用。`finish()` 可重复调用，析构也会自动归还页引用；
结束后继续访问或 decode 会抛 `std::logic_error`。页池和 storage 必须比
生命周期对象活得更久。

步骤执行失败时，本次请求会结束并释放引用，防止留下只有部分 K/V 已写入
的活跃请求。异常路径会尽力等待设备工作完成后再释放。空输入等调用前
参数错误不会改变请求状态，可修正后重试。当前接口是单请求、默认 stream
上的同步流程；批量调度及跨 stream 生命周期管理仍需上层实现。

## 验证

[端到端测试](../tests/sequence_lifecycle_test.cpp) 将 FP32 输出逐步与 CPU
`reference_attention` 对照，检查 prefill、尾页填满、跨块分配、两次 decode、
输出边界、`finish()` 回收，以及写入后输入校验失败和容量不足时的清理。
没有 NVIDIA GPU 时该用例跳过，其他 CPU 元数据测试仍运行。

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure -R '^kvflux_sequence_lifecycle_tests$'
./build-cuda/kvflux_sequence_lifecycle_demo
```

# v2 Milestone 10：Simplified PagedAttention

`paged_attention` 接受设备端 FP32 Q 和 `SequenceState`，直接从
`PagedKVStorage` 的物理页读取 K/V 并写出设备端 FP32 O。K/V 可为 FP32、
FP16 或 BF16；半精度值在计算时转为 FP32。计算过程不生成连续 K/V：

```text
Q[token][query_head][dim]
  + block_table[logical_token / block_size]
  → K/V[physical_block][kv_head][logical_token % block_size][dim]
  → scaled causal attention → O[token][query_head][dim]
```

每个输出元素由一个 CUDA 线程计算，顺序遍历所有可见的 KV token。
线程直接查 block table 定位物理页，并用在线 softmax 维护最大 score、
分母及加权 V，因此不需要 gather 缓冲或 score 矩阵。重复的 QK 点积和
double 精度累加是为了让实现容易检查；此阶段不追求吞吐。

接口见 [paged_attention.h](../include/kvflux/v2/paged_attention.h)。这是单请求
接口，Q 代表 K/V 序列末尾的 `query_tokens` 个 token。整段 prefill 传入
`query_tokens == sequence.num_tokens()`；单步 decode 传入 `query_tokens == 1`。
因果 mask 使第 `i` 个 Q 只访问位置不大于
`sequence.num_tokens() - query_tokens + i` 的 K/V。`query_heads` 必须是
KV 头数的整数倍，相邻 Q 头共用同一 KV 头。

目前每次调用会上传 block table，在默认 stream 上启动 kernel，并同步返回。
调用前必须写完所有有效 K/V；调用期间请求及其物理页必须保持存活。
Q 和 O 必须是存储所在 GPU 上足够大的独立缓冲，彼此及缓存均不可重叠。

## 正确性验收

[GPU 测试](../tests/paged_attention_test.cu) 使用非连续物理页
`[5, 1, 7]`、10 个 token 和只有两个有效 token 的尾页，执行两条路径：

```text
Reference: paged KV → read_paged_kv gather → 连续 K/V → CPU reference_attention
KVFlux:    paged KV + block table + Q → CUDA direct paged_attention
```

测试先逐元素核对 gather 的 K/V 与原始写入值，再比较两条路径的输出。
覆盖 FP32、FP16、BF16、全序列 prefill、末尾三个 Q 和单 token decode，
并检查 `max_error < 1e-5`。未写入的页和尾部槽位预填为 NaN，帮助发现
错误访问；输出末尾的哨兵值用于检查越界写。还检查形状、设备指针、
缓冲重叠和错误页池的拒绝行为。

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure -R '^kvflux_paged_attention_tests$'
```

该测试需要可用的 NVIDIA GPU；没有设备时 CTest 会显示 `Skipped`。

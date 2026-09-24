# v2 Milestone 9：Reference Attention

`reference_attention` 是 CPU 上的连续 FP32 正确性基准。输入 Q 为
`[query_tokens][query_heads][head_size]`，K/V 为
`[kv_tokens][kv_heads][head_size]`；返回值与 Q 同形状。计算顺序是
`QKᵀ → /sqrt(head_size) → causal mask → softmax → V`。每行 softmax
先减去最大 score，点积、归一化与输出累加使用 double，最后转回 float。
实现见 [reference_attention.h](../include/kvflux/v2/reference_attention.h)
和 [reference_attention.cpp](../src/v2/reference_attention.cpp)。

这是单请求接口。Q 对齐 K/V 末尾：第 `i` 个 Q 对应的绝对 token 位置为
`kv_tokens - query_tokens + i`，只可看到不晚于该位置的 K/V。因此
`query_tokens == kv_tokens` 是 prefill，`query_tokens == 1` 是最后一步 decode。
`query_heads` 必须为 `kv_heads` 的整数倍；每组相邻 Q 头共用一个 KV 头。
输入只接受有限 FP32 数值，FP16/BF16 输入应先转换为 FP32。

```cpp
#include "kvflux/v2/reference_attention.h"

kvflux::v2::AttentionShape shape{query_tokens, kv_tokens,
                                 query_heads, kv_heads, head_size};
auto expected = kvflux::v2::reference_attention(q, contiguous_k, contiguous_v, shape);

float max_error = 0.0f;
for (std::size_t i = 0; i < expected.size(); ++i) {
    max_error = std::max(max_error, std::abs(paged_output[i] - expected[i]));
}
// 例如 FP32 路径可从 1e-5f 开始设容差，再按实际 kernel 误差调整。
const float tolerance = 1e-5f;
assert(max_error < tolerance);
```

Milestone 10 的 [PagedAttention kernel](v2_paged_attention.md) 直接按 Block Table
访问页中的 K/V，其输出拷回 CPU 后与此结果比较。[Paged KV Read](v2_paged_kv_read.md)
可先把页中的 K/V 读成连续设备缓冲，再拷回并转换为 FP32，用于检查
输入内容；它本身不执行 attention。此基准以正确性为目标，不做性能优化。

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

CPU 测试覆盖 causal mask、缩放、decode 位置、GQA 头映射、大 logits 下的
稳定 softmax，以及输入形状检查。

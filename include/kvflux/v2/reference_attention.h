#pragma once

#include <cstddef>
#include <vector>

namespace kvflux::v2 {

// 单请求的连续张量形状。Q/O: [query_tokens][query_heads][head_size]；
// K/V: [kv_tokens][kv_heads][head_size]，最后一维连续。
struct AttentionShape {
    std::size_t query_tokens;
    std::size_t kv_tokens;
    std::size_t query_heads;
    std::size_t kv_heads;
    std::size_t head_size;
};

// CPU FP32 正确性基准：softmax((QK^T / sqrt(head_size)) + causal_mask) V。
// Q 对齐 K/V 的末尾：query token i 的绝对位置是 kv_tokens-query_tokens+i。
// 因此既可用于整段 prefill，也可用于只传末尾一个或多个 Q 的 decode。
// query_heads 必须是 kv_heads 的整数倍；相邻的 query_heads/kv_heads 个 Q 头
// 共用一个 KV 头。输入必须是有限数，输出按 Q 的布局排列。
// 半精度 K/V 需先转换为 FP32；此函数不读取设备内存或分页表。
std::vector<float> reference_attention(const std::vector<float>& query,
                                       const std::vector<float>& key,
                                       const std::vector<float>& value,
                                       const AttentionShape& shape);

} // namespace kvflux::v2

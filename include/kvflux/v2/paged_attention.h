#pragma once

#include "kvflux/v2/paged_kv_storage.h"

#include <cstddef>
#include <vector>

namespace kvflux::v2 {

// 单请求的简化 CUDA PagedAttention。Q/O 为设备端 FP32，布局均为
// [query_tokens][query_heads][head_size]；K/V 从 storage 的物理页直接读取。
// Q 对齐 sequence 的末尾：第 i 个 Q 只能看见位置 <=
// sequence.num_tokens()-query_tokens+i 的 K/V。query_heads 必须是 KV 头数的
// 整数倍，相邻的一组 Q 头共享一个 KV 头。
//
// 调用前须写完所有有效 K/V，且 query 与 output 是 storage 所在 GPU 上足够大、
// 互不重叠的独立缓冲；它们不能与缓存重叠。输入数据须为有限数。
// 空 Q 不访问指针。每次调用上传 block table，在默认 stream 执行并同步返回。
// 调用期间 sequence 和物理页必须保持存活，不得并发修改。
void paged_attention(const PagedKVStorage& storage, const SequenceState& sequence,
                     const float* query_device, float* output_device,
                     std::size_t query_tokens, std::size_t query_heads);

// batch decode：每个请求恰有一个末尾 Q，Q/O 均为
// [num_sequences][query_heads][head_size] 的设备端 FP32 连续数组。
// 请求顺序就是 batch 行号；各请求长度可不同，但均须非空且使用 storage 的页池。
// 内部上传行主序 [num_sequences][max_blocks_per_sequence] block table 和
// sequence_lengths，然后由 GPU 用 (request, logical_block) 查物理页。
// 空 batch 不访问指针。其他指针、同步及生命周期约束同 paged_attention。
void paged_attention_batch(const PagedKVStorage& storage,
                           const std::vector<const SequenceState*>& sequences,
                           const float* query_device, float* output_device,
                           std::size_t query_heads);

} // namespace kvflux::v2

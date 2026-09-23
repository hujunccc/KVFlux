#pragma once

#include "kvflux/v2/paged_kv_storage.h"

#include <vector>

namespace kvflux::v2 {

// 第 i 项是请求逻辑块 i 的物理块编号。请求必须保持存活，直至读取完成。
using ReadBlockTable = std::vector<PhysicalBlockID>;
ReadBlockTable build_read_block_table(const SequenceState& sequence);

// 按请求逻辑 token 顺序，将页中的 K/V 收集到设备端输出：
// [num_tokens][num_kv_heads][head_size]，dtype 与 storage 一致。
// 每次调用上传 block table、执行 kernel，并等待默认 stream 完成。
// 输出必须是 storage 所在 GPU 上足够大的独立设备缓冲，且不能与缓存重叠。
// 空请求不访问输出指针；非空请求拒绝错误页池、空指针和错误设备指针。
void read_paged_kv(const PagedKVStorage& storage, const SequenceState& sequence,
                   void* key_out_device, void* value_out_device);

} // namespace kvflux::v2

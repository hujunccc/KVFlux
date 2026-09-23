#pragma once

#include "kvflux/v2/sequence_state.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kvflux::v2 {

// 固定宽度、按 batch 顺序排列的 token 槽位编号，可作为后续 GPU 输入缓冲区。
using PhysicalSlot = std::uint64_t;
using SlotMapping = std::vector<PhysicalSlot>;

struct TokenWrite {
    const SequenceState* sequence; // 只借用；调用期间必须保持存活。
    std::size_t token_position;     // 必须已由 append_token(s) 预留。
};

// 控制面将每个待写 token 转为 physical_block * block_size + offset。
// 输出顺序与 batch 一致；空 batch 返回空 vector。
// 拒绝空请求指针、越界 token、不同物理页池和 slot 算术溢出。
SlotMapping build_slot_mapping(const std::vector<TokenWrite>& batch);

} // namespace kvflux::v2

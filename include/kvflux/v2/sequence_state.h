#pragma once

#include "kvflux/v2/block_table.h"

#include <cstddef>
#include <cstdint>

namespace kvflux::v2 {

using RequestID = std::uint64_t;

struct TokenLocation {
    PhysicalBlockID physical_block;
    std::size_t offset_in_block;
};

// 一个请求的逻辑 KV 地址空间。只管理 token 数和物理页引用，不保存 KV 数据。
// 池必须比请求活得更久；与 PhysicalBlockPool 一样，本类由调用方保证单线程访问。
class SequenceState {
public:
    SequenceState(RequestID request_id, PhysicalBlockPool& pool) noexcept;

    SequenceState(const SequenceState&) = delete;
    SequenceState& operator=(const SequenceState&) = delete;
    SequenceState(SequenceState&& other) noexcept;
    SequenceState& operator=(SequenceState&& other) noexcept;

    RequestID request_id() const noexcept { return request_id_; }
    std::size_t num_tokens() const noexcept { return num_tokens_; }
    std::size_t block_size() const noexcept { return block_size_; }
    const BlockTable& block_table() const noexcept { return block_table_; }
    std::size_t num_allocated_blocks() const noexcept { return block_table_.size(); }
    // 空请求返回 0；满块返回 block_size，而不是 0。
    std::size_t last_block_num_tokens() const noexcept;

    // 为新增 token 预留逻辑位置和物理页，不写入实际 K/V 内容。
    // 任何分配失败都会归还本次新页，请求状态保持原样。
    void append_tokens(std::size_t count);
    // decode 一步：追加一个 token，并返回它的物理页和块内偏移。
    // 尾块未满时复用该页；跨过块边界时才申请新页。
    TokenLocation append_token();

    // 查询本请求的逻辑块和 token 对应的物理页；越界抛 std::out_of_range。
    PhysicalBlockID physical_block_id(std::size_t logical_block) const;
    TokenLocation token_location(std::size_t token_index) const;

private:
    RequestID request_id_;
    std::size_t num_tokens_ = 0;
    std::size_t block_size_;
    BlockTable block_table_;
};

} // namespace kvflux::v2

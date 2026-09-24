#pragma once

#include "kvflux/v2/block_table.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace kvflux::v2 {

using RequestID = std::uint64_t;
class PrefixCache;

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
    // 若尾块是共享且未满的页，拒绝原地追加；改用 append_token_with_copy。
    void append_tokens(std::size_t count);
    // decode 一步：追加一个 token，并返回它的物理页和块内偏移。
    // 尾块未满时复用该页；跨过块边界时才申请新页。
    TokenLocation append_token();

    // 创建共享当前全部逻辑块的新请求，包括未满尾块；每个表项增加一个引用。
    // 后续向共享尾块追加时须走写时复制。源请求保持不变。
    SequenceState fork_shared(RequestID new_request_id) const;

    using BlockCopier = std::function<void(PhysicalBlockHandle source,
                                           PhysicalBlockHandle destination)>;
    // decode 追加：共享且未满的尾块先分配新页并调用 copy 完整复制 K/V，
    // 复制成功后替换本请求的表项，再预留新 token。复制或分配失败时状态不变。
    // 其余情况沿用 append_token，不调用 copy。返回的位置仍需由调用方写入新 K/V。
    TokenLocation append_token_with_copy(const BlockCopier& copy);


    // 查询本请求的逻辑块和 token 对应的物理页；越界抛 std::out_of_range。
    PhysicalBlockID physical_block_id(std::size_t logical_block) const;
    TokenLocation token_location(std::size_t token_index) const;

private:
    friend class PrefixCache;
    // 只允许 PrefixCache 调用：它先校验页池、完整前缀与已发布状态。
    void append_cached_full_block(PhysicalBlockHandle handle);
    PhysicalBlockHandle block_handle(std::size_t logical_block) const;
    RequestID request_id_;
    std::size_t num_tokens_ = 0;
    std::size_t block_size_;
    BlockTable block_table_;
};

} // namespace kvflux::v2

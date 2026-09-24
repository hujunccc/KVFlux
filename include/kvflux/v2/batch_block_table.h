#pragma once

#include "kvflux/v2/sequence_state.h"

#include <cstddef>
#include <vector>

namespace kvflux::v2 {

// 一次 batch 的只读快照；不持有物理页引用。调用方必须让各 SequenceState 及其
// 物理页存活，且在 GPU 使用快照期间不得修改映射。
class BatchBlockTable {
public:
    // 第 request 行从 request * max_blocks_per_sequence() 开始；短行用
    // invalid_block() 填充。空 batch 的宽度为 0。
    std::size_t num_sequences() const noexcept { return sequence_lengths_.size(); }
    std::size_t max_blocks_per_sequence() const noexcept { return max_blocks_per_sequence_; }
    const std::vector<PhysicalBlockID>& block_table() const noexcept { return block_table_; }
    const std::vector<std::size_t>& sequence_lengths() const noexcept { return sequence_lengths_; }
    static constexpr PhysicalBlockID invalid_block() noexcept {
        return static_cast<PhysicalBlockID>(-1);
    }

    // 仅允许访问有效逻辑块；越界或访问 padding 时抛 std::out_of_range。
    PhysicalBlockID physical_id(std::size_t request, std::size_t logical_block) const;

private:
    friend BatchBlockTable build_batch_block_table(const std::vector<const SequenceState*>&);
    std::size_t max_blocks_per_sequence_ = 0;
    std::vector<PhysicalBlockID> block_table_;
    std::vector<std::size_t> sequence_lengths_;
};

// 按输入顺序生成行主序二维表及对应的 token 长度。所有请求须属于同一个池；
// 空请求可占一行，其整行均为 padding。拒绝空指针及不同页池。
BatchBlockTable build_batch_block_table(const std::vector<const SequenceState*>& sequences);

} // namespace kvflux::v2

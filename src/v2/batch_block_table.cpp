#include "kvflux/v2/batch_block_table.h"

#include <limits>
#include <stdexcept>

namespace kvflux::v2 {

PhysicalBlockID BatchBlockTable::physical_id(std::size_t request,
                                             std::size_t logical_block) const {
    if (request >= num_sequences() || logical_block >= max_blocks_per_sequence_) {
        throw std::out_of_range("batch block table index out of range");
    }
    const auto block = block_table_[request * max_blocks_per_sequence_ + logical_block];
    if (block == invalid_block()) throw std::out_of_range("batch block table padding");
    return block;
}

BatchBlockTable build_batch_block_table(const std::vector<const SequenceState*>& sequences) {
    BatchBlockTable result;
    if (sequences.empty()) return result;
    if (!sequences.front()) throw std::invalid_argument("null batch sequence");

    result.sequence_lengths_.reserve(sequences.size());
    for (const auto* sequence : sequences) {
        if (!sequence) throw std::invalid_argument("null batch sequence");
        if (!sequence->block_table().shares_pool_with(sequences.front()->block_table())) {
            throw std::invalid_argument("batch sequences must use one physical pool");
        }
        result.sequence_lengths_.push_back(sequence->num_tokens());
        if (sequence->num_allocated_blocks() > result.max_blocks_per_sequence_) {
            result.max_blocks_per_sequence_ = sequence->num_allocated_blocks();
        }
    }
    if (result.max_blocks_per_sequence_ >
        std::numeric_limits<std::size_t>::max() / sequences.size()) {
        throw std::overflow_error("batch block table size overflow");
    }
    result.block_table_.assign(sequences.size() * result.max_blocks_per_sequence_,
                               BatchBlockTable::invalid_block());
    for (std::size_t request = 0; request < sequences.size(); ++request) {
        // 只复制物理编号；原 SequenceState 仍负责持有并最终归还物理页引用。
        for (std::size_t logical = 0; logical < sequences[request]->num_allocated_blocks(); ++logical) {
            result.block_table_[request * result.max_blocks_per_sequence_ + logical] =
                sequences[request]->physical_block_id(logical);
        }
    }
    return result;
}

} // namespace kvflux::v2

#include "kvflux/v2/sequence_state.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace kvflux::v2 {

SequenceState::SequenceState(RequestID request_id, PhysicalBlockPool& pool) noexcept
    : request_id_(request_id), block_size_(pool.block_size()), block_table_(pool) {}

SequenceState::SequenceState(SequenceState&& other) noexcept
    : request_id_(other.request_id_),
      num_tokens_(std::exchange(other.num_tokens_, 0)),
      block_size_(other.block_size_),
      block_table_(std::move(other.block_table_)) {}

SequenceState& SequenceState::operator=(SequenceState&& other) noexcept {
    if (this == &other) return *this;
    block_table_ = std::move(other.block_table_);
    request_id_ = other.request_id_;
    num_tokens_ = std::exchange(other.num_tokens_, 0);
    block_size_ = other.block_size_;
    return *this;
}

std::size_t SequenceState::last_block_num_tokens() const noexcept {
    return num_tokens_ == 0 ? 0 : (num_tokens_ - 1) % block_size_ + 1;
}

void SequenceState::append_tokens(std::size_t count) {
    if (count == 0) return;
    if (count > std::numeric_limits<std::size_t>::max() - num_tokens_) {
        throw std::overflow_error("sequence token count overflow");
    }

    const auto next_tokens = num_tokens_ + count;
    const auto needed_blocks = next_tokens / block_size_ + (next_tokens % block_size_ != 0);
    const auto original_blocks = block_table_.size();
    try {
        while (block_table_.size() < needed_blocks) block_table_.append_new();
    } catch (...) {
        while (block_table_.size() > original_blocks) block_table_.pop_back();
        throw;
    }
    num_tokens_ = next_tokens;
}

TokenLocation SequenceState::append_token() {
    const auto token_index = num_tokens_;
    append_tokens(1);
    return token_location(token_index);
}

void SequenceState::append_cached_full_block(PhysicalBlockHandle handle) {
    if (num_tokens_ % block_size_ != 0 ||
        block_size_ > std::numeric_limits<std::size_t>::max() - num_tokens_) {
        throw std::invalid_argument("cached block requires a complete preceding prefix");
    }
    block_table_.append_existing(handle); // 失败时 token 数和表均保持不变。
    num_tokens_ += block_size_;
}

PhysicalBlockID SequenceState::physical_block_id(std::size_t logical_block) const {
    return block_table_.physical_id(logical_block);
}

PhysicalBlockHandle SequenceState::block_handle(std::size_t logical_block) const {
    return block_table_.handle(logical_block);
}

TokenLocation SequenceState::token_location(std::size_t token_index) const {
    if (token_index >= num_tokens_) throw std::out_of_range("token index is outside sequence");
    return {physical_block_id(token_index / block_size_), token_index % block_size_};
}

} // namespace kvflux::v2

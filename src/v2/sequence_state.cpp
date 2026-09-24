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
    if (num_tokens_ % block_size_ != 0 &&
        block_table_.ref_count(block_table_.size() - 1) > 1) {
        throw std::logic_error("shared partial block requires copy-on-write");
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

SequenceState SequenceState::fork_shared(RequestID new_request_id) const {
    SequenceState fork(new_request_id, block_table_.pool());
    // append_shared 逐项 retain；如果中途失败，fork 的析构会归还已取得的引用。
    for (std::size_t logical = 0; logical < block_table_.size(); ++logical) {
        fork.block_table_.append_shared(block_table_, logical);
    }
    fork.num_tokens_ = num_tokens_;
    return fork;
}

TokenLocation SequenceState::append_token_with_copy(const BlockCopier& copy) {
    if (num_tokens_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("sequence token count overflow");
    }
    const auto offset = num_tokens_ % block_size_;
    if (offset == 0 || block_table_.ref_count(block_table_.size() - 1) == 1) {
        return append_token();
    }
    if (!copy) throw std::invalid_argument("copy callback is required for shared partial block");

    const auto old_handle = block_table_.handle(block_table_.size() - 1);
    auto& pool = block_table_.pool();
    const auto new_handle = pool.allocate();
    try {
        copy(old_handle, new_handle); // 复制完成前不能改变请求映射或 token 数。
        block_table_.replace_last_owned(new_handle);
    } catch (...) {
        pool.free(new_handle);
        throw;
    }
    ++num_tokens_;
    return {new_handle.id, offset};
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

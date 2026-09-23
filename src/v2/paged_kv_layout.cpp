#include "kvflux/v2/paged_kv_layout.h"

#include <limits>
#include <stdexcept>

namespace kvflux::v2 {
namespace {

std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("paged KV layout size overflow");
    }
    return a * b;
}

std::size_t bytes_for_dtype(DType dtype) {
    switch (dtype) {
    case DType::Float16:
    case DType::BFloat16: return 2;
    case DType::Float32: return 4;
    }
    throw std::invalid_argument("unsupported paged KV dtype");
}

} // namespace

PagedKVLayout::PagedKVLayout(std::size_t num_blocks, std::size_t num_kv_heads,
                             std::size_t block_size, std::size_t head_size, DType dtype)
    : num_blocks_(num_blocks), num_kv_heads_(num_kv_heads), block_size_(block_size),
      head_size_(head_size), dtype_(dtype), element_bytes_(bytes_for_dtype(dtype)),
      page_bytes_(0), cache_bytes_(0), total_bytes_(0), slot_count_(0) {
    if (!num_blocks || !num_kv_heads || !block_size || !head_size) {
        throw std::invalid_argument("paged KV dimensions must be positive");
    }
    page_bytes_ = checked_multiply(
        checked_multiply(checked_multiply(num_kv_heads, block_size), head_size), element_bytes_);
    cache_bytes_ = checked_multiply(num_blocks, page_bytes_);
    total_bytes_ = checked_multiply(2, cache_bytes_);
    slot_count_ = checked_multiply(num_blocks, block_size);
}

std::size_t PagedKVLayout::page_byte_offset(KVKind kind, std::size_t block) const {
    if (kind != KVKind::Key && kind != KVKind::Value) {
        throw std::invalid_argument("invalid KV kind");
    }
    if (block >= num_blocks_) throw std::out_of_range("physical block out of range");
    return (kind == KVKind::Value ? cache_bytes_ : 0) + block * page_bytes_;
}

std::size_t PagedKVLayout::byte_offset(KVKind kind, std::size_t block,
                                       std::size_t head, std::size_t token,
                                       std::size_t dimension) const {
    const auto page = page_byte_offset(kind, block);
    if (head >= num_kv_heads_ || token >= block_size_ || dimension >= head_size_) {
        throw std::out_of_range("paged KV coordinate out of range");
    }
    const auto element = ((head * block_size_ + token) * head_size_ + dimension);
    return page + element * element_bytes_;
}

std::size_t PagedKVLayout::slot_byte_offset(KVKind kind, std::uint64_t slot,
                                            std::size_t head, std::size_t dimension) const {
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "paged KV slot requires size_t to fit in uint64_t");
    if (slot >= slot_count_) throw std::out_of_range("physical slot out of range");
    return byte_offset(kind, static_cast<std::size_t>(slot / block_size_), head,
                       static_cast<std::size_t>(slot % block_size_), dimension);
}

} // namespace kvflux::v2

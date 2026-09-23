#pragma once

#include "kvflux/kv_layout.h"

#include <cstddef>
#include <cstdint>

namespace kvflux::v2 {

// 单层 K/V 布局：K[num_blocks][num_kv_heads][block_size][head_size]，
// 随后是同形状的 V。最后一维连续，无 padding。
class PagedKVLayout {
public:
    PagedKVLayout(std::size_t num_blocks, std::size_t num_kv_heads,
                  std::size_t block_size, std::size_t head_size, DType dtype);

    std::size_t num_blocks() const noexcept { return num_blocks_; }
    std::size_t num_kv_heads() const noexcept { return num_kv_heads_; }
    std::size_t block_size() const noexcept { return block_size_; }
    std::size_t head_size() const noexcept { return head_size_; }
    DType dtype() const noexcept { return dtype_; }
    std::size_t element_bytes() const noexcept { return element_bytes_; }
    std::size_t page_bytes() const noexcept { return page_bytes_; }
    std::size_t cache_bytes() const noexcept { return cache_bytes_; }
    std::size_t total_bytes() const noexcept { return total_bytes_; }
    std::size_t slot_count() const noexcept { return slot_count_; }

    // 相对整个 K+V 分配起点的字节偏移；所有坐标都经过边界检查。
    std::size_t page_byte_offset(KVKind kind, std::size_t block) const;
    std::size_t byte_offset(KVKind kind, std::size_t block, std::size_t head,
                            std::size_t token, std::size_t dimension) const;
    // slot = physical_block * block_size + token；仍需提供 head 和 dimension。
    std::size_t slot_byte_offset(KVKind kind, std::uint64_t slot,
                                 std::size_t head, std::size_t dimension) const;

private:
    std::size_t num_blocks_, num_kv_heads_, block_size_, head_size_;
    DType dtype_;
    std::size_t element_bytes_, page_bytes_, cache_bytes_, total_bytes_, slot_count_;
};

} // namespace kvflux::v2

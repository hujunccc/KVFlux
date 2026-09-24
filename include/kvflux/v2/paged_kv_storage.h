#pragma once

#include "kvflux/gpu_memory_pool.h"
#include "kvflux/physical_block_pool.h"
#include "kvflux/v2/paged_kv_layout.h"
#include "kvflux/v2/slot_mapping.h"

#include <cstddef>

namespace kvflux::v2 {

// 单层、单 GPU 的实际 K/V 显存。使用 v1 GpuMemoryPool 做一次性底层分配，
// 使用 v2 PhysicalBlockPool 作为唯一的物理页编号/引用所有者。
class PagedKVStorage {
public:
    PagedKVStorage(PhysicalBlockPool& pool, std::size_t num_kv_heads,
                   std::size_t head_size, DType dtype, int device = 0);

    PagedKVStorage(const PagedKVStorage&) = delete;
    PagedKVStorage& operator=(const PagedKVStorage&) = delete;

    const PagedKVLayout& layout() const noexcept { return layout_; }
    int device() const noexcept { return backing_.device(); }
    bool uses_pool(const BlockTable& table) const noexcept { return table.uses_pool(pool_); }
    void* key_base() const;
    void* value_base() const;

    // 带句柄的查询验证 v2 页引用；返回 device pointer，CPU 不能解引用。
    // 地址查询不代表内容已初始化，也不授予对共享页的写权限。
    void* page_address(KVKind kind, PhysicalBlockHandle handle) const;
    void* element_address(KVKind kind, PhysicalBlockHandle handle,
                          std::size_t head, std::size_t token, std::size_t dimension) const;
    // slot_mapping 的编号只检查范围，不持有页引用；调用方需保持页存活直到 GPU 完成。
    void* slot_address(KVKind kind, PhysicalSlot slot,
                       std::size_t head, std::size_t dimension) const;

    // 将源页的完整 K/V 内容复制到私有的新页，并等待默认 stream 完成。
    // 仅供写时复制等场景使用；调用期间两页均须保持引用，且不能并发修改。
    void copy_block(PhysicalBlockHandle source, PhysicalBlockHandle destination) const;

private:
    PhysicalBlockPool& pool_; // pool 必须比 storage 活得更久。
    PagedKVLayout layout_;
    GpuMemoryPool backing_; // 内部只使用其原始设备地址接口，分配归属由 v2 pool 管理。
};

} // namespace kvflux::v2

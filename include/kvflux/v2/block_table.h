#pragma once

#include "kvflux/physical_block_pool.h"

#include <cstddef>
#include <vector>

namespace kvflux::v2 {

// 一个表描述一条序列的逻辑块到物理块的映射。
// logical block ID 不需要单独的对象：blocks_[i] 的下标 i 就是 logical block ID。
// 表里的每一项持有 PhysicalBlockPool 的一个引用，析构时自动归还。
// 因为引用有所有权，表可以移动但不能普通复制；共享请用 append_shared。
class BlockTable {
public:
    explicit BlockTable(PhysicalBlockPool& pool) noexcept;
    ~BlockTable() noexcept;

    BlockTable(const BlockTable&) = delete;
    BlockTable& operator=(const BlockTable&) = delete;
    BlockTable(BlockTable&& other) noexcept;
    BlockTable& operator=(BlockTable&& other) noexcept;

    // 分配一个新物理块并追加到表尾，返回它的编号。
    // 分配或 vector 扩容失败时，表和引用计数保持原样。
    PhysicalBlockID append_new();
    // 已有持有者继续保留自己的引用；表为该块增加一个独立引用。
    void append_existing(PhysicalBlockHandle handle);
    // 从另一张表共享一个逻辑块；两张表必须使用同一个物理块池。
    void append_shared(const BlockTable& source, std::size_t logical_block);

    // 查询只返回编号，不增加引用。表仍存活时可用于定位存储层的页。
    // logical_block 越界时抛 std::out_of_range。
    PhysicalBlockID physical_id(std::size_t logical_block) const;
    // table[i] 与 physical_id(i) 等价：i 就是逻辑块编号。
    PhysicalBlockID operator[](std::size_t logical_block) const {
        return physical_id(logical_block);
    }
    std::size_t size() const noexcept { return blocks_.size(); }
    bool empty() const noexcept { return blocks_.empty(); }
    // 同一 batch 的 slot 编号只有在共享同一个物理页池时才可合并。
    bool shares_pool_with(const BlockTable& other) const noexcept {
        return pool_ && pool_ == other.pool_;
    }
    bool uses_pool(const PhysicalBlockPool& pool) const noexcept { return pool_ == &pool; }

    // 删除表尾映射并归还它持有的引用；不能在 GPU 仍使用该页时调用。
    void pop_back();
    void clear() noexcept;

private:
    PhysicalBlockPool* pool_; // 池必须比表活得更久；移动后的源对象这里为 nullptr。
    std::vector<PhysicalBlockHandle> blocks_;
};

} // namespace kvflux::v2

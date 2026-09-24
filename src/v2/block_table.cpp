#include "kvflux/v2/block_table.h"

#include <stdexcept>
#include <utility>

namespace kvflux::v2 {

BlockTable::BlockTable(PhysicalBlockPool& pool) noexcept : pool_(&pool) {}

BlockTable::~BlockTable() noexcept { clear(); }

BlockTable::BlockTable(BlockTable&& other) noexcept : pool_(std::exchange(other.pool_, nullptr)) {
    // 用 swap 明确地把条目交给新表；源表变空，不会在析构时重复 free。
    blocks_.swap(other.blocks_);
}

BlockTable& BlockTable::operator=(BlockTable&& other) noexcept {
    if (this == &other) return *this;
    clear(); // 先归还当前表的引用，再接管另一张表的引用。
    pool_ = std::exchange(other.pool_, nullptr);
    blocks_.swap(other.blocks_);
    return *this;
}

PhysicalBlockID BlockTable::append_new() {
    if (!pool_) throw std::logic_error("cannot append to a moved-from block table");
    auto handle = pool_->allocate(); // allocate 给表一个新引用。
    try {
        blocks_.push_back(handle);
    } catch (...) {
        pool_->free(handle); // vector 扩容失败时归还新分配的引用。
        throw;
    }
    return handle.id;
}

void BlockTable::append_existing(PhysicalBlockHandle handle) {
    if (!pool_) throw std::logic_error("cannot append to a moved-from block table");
    pool_->retain(handle); // 表获得自己的引用，原持有者仍需自行 free。
    try {
        blocks_.push_back(handle);
    } catch (...) {
        pool_->free(handle);
        throw;
    }
}

void BlockTable::append_shared(const BlockTable& source, std::size_t logical_block) {
    if (!pool_ || pool_ != source.pool_) {
        throw std::invalid_argument("block tables must use the same physical pool");
    }
    // at 会检查逻辑下标。即使 source 是 *this，也先读取句柄再可能扩容。
    append_existing(source.blocks_.at(logical_block));
}

PhysicalBlockID BlockTable::physical_id(std::size_t logical_block) const {
    // pool_->id 再验证 generation，防止手动误释放后读到已复用的编号。
    const auto handle = blocks_.at(logical_block);
    return pool_->id(handle);
}

PhysicalBlockHandle BlockTable::handle(std::size_t logical_block) const {
    const auto result = blocks_.at(logical_block);
    (void)pool_->id(result); // 校验 generation 和活跃引用。
    return result;
}

std::size_t BlockTable::ref_count(std::size_t logical_block) const {
    return pool_->ref_count(blocks_.at(logical_block));
}

PhysicalBlockPool& BlockTable::pool() const {
    if (!pool_) throw std::logic_error("moved-from block table has no pool");
    return *pool_;
}

void BlockTable::pop_back() {
    if (blocks_.empty()) throw std::out_of_range("block table is empty");
    pool_->free(blocks_.back());
    blocks_.pop_back();
}

void BlockTable::replace_last_owned(PhysicalBlockHandle new_handle) {
    if (blocks_.empty()) throw std::out_of_range("block table is empty");
    (void)pool_->id(new_handle);
    if (new_handle == blocks_.back() || pool_->ref_count(new_handle) != 1 ||
        pool_->is_published(new_handle)) {
        throw std::invalid_argument("replacement block must be private and unpublished");
    }
    const auto old_handle = blocks_.back();
    // free 先完成所有旧句柄校验；随后句柄赋值不会抛异常。这样一旦
    // 归还旧引用失败，调用方仍可安全归还自己持有的新页。
    pool_->free(old_handle);
    blocks_.back() = new_handle; // 所有权转移，不增加引用。
}

void BlockTable::clear() noexcept {
    // 表独占这些“引用”，不是独占物理页；别的表仍可持有同一个页。
    for (auto handle : blocks_) pool_->free(handle);
    blocks_.clear();
}

} // namespace kvflux::v2

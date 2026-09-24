#pragma once

#include "kvflux/block_manager.h"

#include <cstddef>

namespace kvflux {

// GPU KV 缓存中的物理页编号。它是下标，不是 CUDA 指针，也不代表所有权。
// PagedKVStorage 使用这个编号定位 K/V 的设备页地址。
using PhysicalBlockID = std::size_t;

// 复用 v0 的 BlockHandle：id 指向物理页，generation 区分同一页的前后两次分配。
// 持有句柄的调用方必须为每次 allocate/retain 配对调用一次 free。
using PhysicalBlockHandle = BlockHandle;

// v2 的纯元数据层：负责管理可用物理页，不分配显存、不保存 K/V，也不认识请求。
// 内部直接使用 v0 BlockManager 的 free list、引用计数和旧句柄校验。
// 完整页发布后可按前缀复用；零引用缓存页留在空闲缓存队列，容量不足时淘汰。
// 未发布页在最后一个引用释放后直接进入 free list。
// 与 v0 一样，本类是单线程的；跨线程使用时由调用方加锁。
class PhysicalBlockPool {
public:
    // hasher 可注入以验证哈希碰撞；命中时仍会精确比较完整 token 前缀。
    PhysicalBlockPool(std::size_t capacity, std::size_t block_size,
                      BlockManager::Hasher hasher = BlockManager::prefix_hash);

    PhysicalBlockPool(const PhysicalBlockPool&) = delete;
    PhysicalBlockPool& operator=(const PhysicalBlockPool&) = delete;
    PhysicalBlockPool(PhysicalBlockPool&&) = delete;
    PhysicalBlockPool& operator=(PhysicalBlockPool&&) = delete;

    // 返回一个新引用。池满且所有页仍被持有时抛 CapacityError。
    PhysicalBlockHandle allocate();
    // 同一物理页需要多位持有者时增加一个引用；必须额外 free 一次。
    void retain(PhysicalBlockHandle handle);
    // 释放一个引用。最后一个引用释放后，该编号可以被下一次 allocate 复用。
    void free(PhysicalBlockHandle handle);

    // 仅在完整块的 K/V 已写入并对后续读取可见时发布；prefix 包含从请求
    // 开头到该块末尾的全部 token。发布的页引用归零后留在空闲缓存队列中。
    void publish(PhysicalBlockHandle handle, const Tokens& prefix);
    // 命中时返回一个新引用，调用方必须 free 或交给持有者；哈希碰撞不会误命中。
    bool lookup(const Tokens& prefix, PhysicalBlockHandle& out);
    bool is_published(PhysicalBlockHandle handle) const;

    // 只有活跃句柄才能取得可供存储层使用的编号；旧 generation 会被拒绝。
    // 返回值仅用于定位，不延长页的生命周期。
    PhysicalBlockID id(PhysicalBlockHandle handle) const;
    std::size_t ref_count(PhysicalBlockHandle handle) const;
    std::size_t capacity() const noexcept;
    std::size_t block_size() const noexcept;
    // 可立即分配的空闲页数，包含可淘汰的零引用缓存页。
    std::size_t available() const noexcept;

private:
    BlockManager manager_;
};

} // namespace kvflux

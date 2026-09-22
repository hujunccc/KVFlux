#pragma once

#include "kvflux/block_manager.h"
#include "kvflux/kv_layout.h"
#include <vector>

namespace kvflux {

using BlockId = std::size_t;

// 固定容量、单 GPU 的显存池。构造时只申请一次显存，析构时统一释放。
// 分配一个 block 仅修改 v0 管理器的元数据，不会再次调用 cudaMalloc。
class GpuMemoryPool {
public:
    GpuMemoryPool(std::size_t total_blocks, std::size_t block_bytes,
                  std::size_t tokens_per_block, int device = 0);
    // 根据统一布局向下取整预算；不足一个 block 则拒绝初始化。
    GpuMemoryPool(const KVBlockLayout& layout, std::size_t budget_bytes, int device = 0);
    ~GpuMemoryPool() noexcept;
    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;
    GpuMemoryPool(GpuMemoryPool&&) = delete;
    GpuMemoryPool& operator=(GpuMemoryPool&&) = delete;

    std::size_t total_blocks() const noexcept { return manager_.capacity(); }
    std::size_t block_bytes() const noexcept { return block_bytes_; }
    std::size_t allocated_bytes() const noexcept { return allocated_bytes_; }
    int device() const noexcept { return device_; }
    // 可供新请求使用的块 = 从未使用/已归还的块 + 可淘汰的零引用缓存块。
    std::size_t remaining_blocks() const noexcept;
    BlockManager::Stats stats() const noexcept { return manager_.stats(); }

    BlockHandle allocate();
    void retain(BlockHandle h) { manager_.retain(h); }
    void release(BlockHandle h) { manager_.release(h); }
    std::size_t ref_count(BlockHandle h) const { return manager_.ref_count(h); }

    // 原始地址查询只检查 id 范围，不表示拥有该块。主机不能解引用 GPU 指针。
    // 外部 kernel 使用者必须持有引用并自行同步；直接写地址会绕过保护。
    void* device_address(BlockId id) const;
    void* device_address(BlockHandle h) const;

    // 整块、同步、按字节复制；不做 dtype 转换。bytes 必须等于 block_bytes()。
    // 写入只允许独占且未发布的块；读/发布要求当前 generation 已完整写入。
    void write_block(BlockHandle h, const void* host, std::size_t bytes);
    void read_block(BlockHandle h, void* host, std::size_t bytes) const;
    void publish(BlockHandle h, const Tokens& prefix);
    bool lookup(const Tokens& prefix, BlockHandle& out) { return manager_.lookup(prefix, out); }

private:
    BlockManager manager_;
    std::size_t block_bytes_, allocated_bytes_;
    int device_;
    std::vector<bool> initialized_;
    void* base_ = nullptr;
};

} // namespace kvflux

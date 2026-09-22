#pragma once

#include "kvflux/detail/index_lru.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace kvflux {

using Token = std::int32_t;
using Tokens = std::vector<Token>;

// id 是物理槽位；generation 区分同一槽位的不同生命周期，防止误用旧句柄。
struct BlockHandle {
    std::size_t id;
    std::uint64_t generation;
    bool operator==(const BlockHandle& other) const noexcept {
        return id == other.id && generation == other.generation;
    }
};

struct BlockTable {
    // 第 i 项对应序列中第 i 个逻辑块。表的普通复制不会增加引用计数！
    std::vector<BlockHandle> blocks;
    std::size_t token_count = 0;
};

class CapacityError : public std::runtime_error {
public:
    CapacityError() : std::runtime_error("no reclaimable KV block") {}
};

// 单线程元数据管理器。句柄/表只能用于创建它们的实例；引用需显式释放。
// 一个实例对应固定的模型和推理配置，本版缓存键不包含模型身份。
class BlockManager {
public:
    // 可注入哈希函数以测试碰撞；生产使用默认的确定性前缀哈希。
    using Hasher = std::function<std::uint64_t(const Tokens&)>;
    struct Stats {
        std::size_t capacity, free, active, cached_idle;
    };

    explicit BlockManager(std::size_t capacity, std::size_t block_size,
                          Hasher hasher = prefix_hash);
    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;
    BlockManager(BlockManager&&) = delete;
    BlockManager& operator=(BlockManager&&) = delete;

    // 底层生命周期：allocate 返回一个引用，retain/release 成对使用。
    BlockHandle allocate();
    void retain(BlockHandle handle);
    void release(BlockHandle handle);
    std::size_t ref_count(BlockHandle handle) const;
    bool is_published(BlockHandle handle) const;

    // 发布一个完整块：prefix 包含从序列开头到该块结尾的全部 token。
    // 只存元数据；接入真实后端时，必须先写完 K/V，再发布。
    void publish(BlockHandle handle, const Tokens& prefix);
    // 查找成功就增加引用，返回 true；失败时不修改 out。
    bool lookup(const Tokens& prefix, BlockHandle& out);

    // 教学用序列接口：完整块自动发布，尾部不足一块则保持私有。
    // 接入异步计算后端时应使用 allocate/publish 分离的底层接口。
    BlockTable acquire(const Tokens& tokens);
    BlockTable share(const BlockTable& table);
    void release(BlockTable& table);
    // 返回借用句柄，不新增引用。物理块内偏移为 token_index % block_size()。
    BlockHandle block_at(const BlockTable& table, std::size_t token_index) const;

    Stats stats() const noexcept;
    std::size_t capacity() const noexcept { return slots_.size(); }
    std::size_t block_size() const noexcept { return block_size_; }
    static std::uint64_t prefix_hash(const Tokens& prefix) noexcept;

private:
    static constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
    struct Slot {
        std::uint64_t generation = 0;
        std::size_t refs = 0;
        bool free = true;
        bool cached = false;
        Tokens prefix;
        std::uint64_t hash = 0;
        std::size_t next_free = none;
    };
    Slot& checked(BlockHandle handle);
    const Slot& checked(BlockHandle handle) const;
    void validate(const BlockTable& table) const;
    void unlink_idle(std::size_t id) noexcept;
    void append_idle(std::size_t id) noexcept;
    void erase_cache(std::size_t id);

    std::vector<Slot> slots_;
    std::size_t block_size_;
    Hasher hasher_;
    std::size_t free_head_ = none;
    detail::IndexLru idle_lru_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> index_;
};

} // namespace kvflux

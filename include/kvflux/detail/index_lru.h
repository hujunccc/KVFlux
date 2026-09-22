#pragma once
#include <cstddef>
#include <limits>
#include <vector>

namespace kvflux::detail {
// 同一个基于下标的双向链表，供 v0 空闲缓存和 v1 GPU residency 策略复用。
// 调用者保证 id < capacity；每个 id 最多出现一次，节点数组只在构造时分配。
class IndexLru {
public:
    static constexpr auto none = std::numeric_limits<std::size_t>::max();
    explicit IndexLru(std::size_t capacity) : nodes_(capacity) {}
    std::size_t oldest() const noexcept { return oldest_; }
    std::size_t next(std::size_t id) const noexcept { return nodes_[id].next; }
    void erase(std::size_t id) noexcept {
        auto& n = nodes_[id];
        if (!n.present) return;
        if (n.prev == none) oldest_ = n.next; else nodes_[n.prev].next = n.next;
        if (n.next == none) newest_ = n.prev; else nodes_[n.next].prev = n.prev;
        n = Node{};
    }
    void touch(std::size_t id) noexcept {
        erase(id);
        auto& n = nodes_[id];
        n.present = true;
        n.prev = newest_;
        if (newest_ == none) oldest_ = id; else nodes_[newest_].next = id;
        newest_ = id;
    }
private:
    struct Node { std::size_t prev = none, next = none; bool present = false; };
    std::vector<Node> nodes_;
    std::size_t oldest_ = none, newest_ = none;
};
} // namespace kvflux::detail

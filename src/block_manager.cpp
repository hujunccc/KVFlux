#include "kvflux/block_manager.h"

#include <algorithm>
#include <utility>

namespace kvflux {

BlockManager::BlockManager(std::size_t capacity, std::size_t block_size, Hasher hasher)
    : slots_(capacity), block_size_(block_size), hasher_(std::move(hasher)), idle_lru_(capacity) {
    if (capacity == 0 || block_size == 0 || !hasher_) {
        throw std::invalid_argument("capacity, block size and hasher must be valid");
    }
    // 初始化单向 free list，分配时只需要弹出表头，无须扫描全部槽位。
    for (std::size_t i = 0; i < capacity; ++i) {
        slots_[i].next_free = i + 1 < capacity ? i + 1 : none;
    }
    free_head_ = 0;
}

const BlockManager::Slot& BlockManager::checked(BlockHandle h) const {
    if (h.id >= slots_.size() || slots_[h.id].free ||
        slots_[h.id].generation != h.generation) {
        throw std::invalid_argument("invalid or stale block handle");
    }
    return slots_[h.id];
}

BlockManager::Slot& BlockManager::checked(BlockHandle h) {
    return const_cast<Slot&>(static_cast<const BlockManager&>(*this).checked(h));
}

void BlockManager::unlink_idle(std::size_t id) noexcept { idle_lru_.erase(id); }
void BlockManager::append_idle(std::size_t id) noexcept { idle_lru_.touch(id); }

void BlockManager::erase_cache(std::size_t id) {
    auto& s = slots_[id];
    auto it = index_.find(s.hash);
    auto& bucket = it->second;
    bucket.erase(std::find(bucket.begin(), bucket.end(), id));
    if (bucket.empty()) index_.erase(it);
    s.cached = false;
    s.prefix.clear();
}

BlockHandle BlockManager::allocate() {
    const auto id = free_head_ != none ? free_head_ : idle_lru_.oldest();
    if (id == none) throw CapacityError();
    auto& s = slots_[id];
    if (s.generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("block generation exhausted");
    }
    if (s.free) {
        free_head_ = s.next_free;
        s.next_free = none;
    } else {
        // 只有 refs==0 的缓存块才进入 LRU，因此不会淘汰正在使用的块。
        unlink_idle(id);
        erase_cache(id);
    }
    s.free = false;
    s.refs = 1;
    ++s.generation;
    return {id, s.generation};
}

void BlockManager::retain(BlockHandle h) {
    auto& s = checked(h);
    if (s.refs == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("reference count exhausted");
    }
    if (s.refs == 0) unlink_idle(h.id);
    ++s.refs;
}

void BlockManager::release(BlockHandle h) {
    auto& s = checked(h);
    if (s.refs == 0) throw std::invalid_argument("block already released");
    if (--s.refs != 0) return;
    if (s.cached) {
        append_idle(h.id);
    } else {
        s.free = true;
        s.next_free = free_head_;
        free_head_ = h.id;
    }
}

std::size_t BlockManager::ref_count(BlockHandle h) const { return checked(h).refs; }
bool BlockManager::is_published(BlockHandle h) const { return checked(h).cached; }

void BlockManager::publish(BlockHandle h, const Tokens& prefix) {
    auto& s = checked(h);
    if (s.refs != 1 || s.cached || prefix.empty() || prefix.size() % block_size_ != 0) {
        throw std::invalid_argument("publish requires a private block and a full prefix");
    }
    const auto hash = hasher_(prefix);
    const auto existing = index_.find(hash);
    if (existing != index_.end()) {
        for (auto id : existing->second) {
            if (slots_[id].prefix == prefix) {
                throw std::invalid_argument("prefix already published; use lookup");
            }
        }
    }
    // 先完成可能失败的分配，再修改槽位状态，避免半发布的数据被查到。
    Tokens copy = prefix;
    auto inserted = index_.try_emplace(hash);
    try {
        inserted.first->second.push_back(h.id);
    } catch (...) {
        if (inserted.second) index_.erase(inserted.first);
        throw;
    }
    s.prefix = std::move(copy);
    s.hash = hash;
    s.cached = true;
}

bool BlockManager::lookup(const Tokens& prefix, BlockHandle& out) {
    if (prefix.empty() || prefix.size() % block_size_ != 0) return false;
    auto it = index_.find(hasher_(prefix));
    if (it == index_.end()) return false;
    for (auto id : it->second) {
        const auto& s = slots_[id];
        // 哈希相同不代表 token 相同；必须比较完整前缀，保证碰撞不会误复用。
        if (s.prefix == prefix) {
            BlockHandle h{id, s.generation};
            retain(h);
            out = h;
            return true;
        }
    }
    return false;
}

BlockTable BlockManager::acquire(const Tokens& tokens) {
    BlockTable table;
    table.token_count = tokens.size();
    table.blocks.reserve(tokens.size() / block_size_ + (tokens.size() % block_size_ != 0));
    Tokens prefix;
    prefix.reserve(tokens.size());
    try {
        for (std::size_t offset = 0; offset < tokens.size();) {
            const auto count = std::min(block_size_, tokens.size() - offset);
            prefix.insert(prefix.end(), tokens.begin() + offset, tokens.begin() + offset + count);
            BlockHandle h{};
            const bool full = count == block_size_;
            if (full && lookup(prefix, h)) {
                table.blocks.push_back(h);
            } else {
                h = allocate();
                table.blocks.push_back(h); // 已 reserve，不会因扩容而丢失引用。
                if (full) publish(h, prefix);
            }
            offset += count;
        }
    } catch (...) {
        // 失败时表尚未完整，不能使用要求完整表的 release(table)。
        // 逐个回滚已取得的引用；允许保留已完成的缓存。
        for (auto h : table.blocks) release(h);
        throw;
    }
    return table;
}

void BlockManager::validate(const BlockTable& table) const {
    const auto expected = table.token_count / block_size_ + (table.token_count % block_size_ != 0);
    if (expected != table.blocks.size()) throw std::invalid_argument("invalid block table size");
    for (std::size_t i = 0; i < table.blocks.size(); ++i) {
        if (checked(table.blocks[i]).refs == 0) throw std::invalid_argument("unowned block");
        for (std::size_t j = 0; j < i; ++j) {
            if (table.blocks[i] == table.blocks[j]) throw std::invalid_argument("duplicate block");
        }
    }
}

BlockTable BlockManager::share(const BlockTable& table) {
    validate(table);
    BlockTable copy = table;
    std::size_t retained = 0;
    try {
        for (auto h : copy.blocks) { retain(h); ++retained; }
    } catch (...) {
        for (std::size_t i = 0; i < retained; ++i) release(copy.blocks[i]);
        throw;
    }
    return copy;
}

void BlockManager::release(BlockTable& table) {
    validate(table); // 先整表验证，避免遇到坏句柄时只释放了一半。
    for (auto h : table.blocks) release(h);
    table.blocks.clear();
    table.token_count = 0;
}

BlockHandle BlockManager::block_at(const BlockTable& table, std::size_t token_index) const {
    if (token_index >= table.token_count) throw std::out_of_range("token index out of range");
    const auto h = table.blocks.at(token_index / block_size_);
    if (checked(h).refs == 0) throw std::invalid_argument("unowned block");
    return h;
}

BlockManager::Stats BlockManager::stats() const noexcept {
    Stats result{slots_.size(), 0, 0, 0};
    for (const auto& s : slots_) {
        if (s.free) ++result.free;
        else if (s.refs != 0) ++result.active;
        else ++result.cached_idle;
    }
    return result;
}

std::uint64_t BlockManager::prefix_hash(const Tokens& prefix) noexcept {
    // FNV-1a：将有符号 token 转为 32 位无符号数，按固定小端字节顺序混合。
    // 不依赖机器字节序；不是密码学哈希，正确性由 lookup 的精确比较保证。
    std::uint64_t hash = 14695981039346656037ULL;
    for (Token token : prefix) {
        auto value = static_cast<std::uint32_t>(token);
        for (unsigned byte = 0; byte < 4; ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

} // namespace kvflux

#include "kvflux/v2/prefix_cache.h"

#include <stdexcept>

namespace kvflux::v2 {

std::size_t PrefixCache::attach_cached_prefix(SequenceState& sequence, const Tokens& tokens) {
    if (!sequence.block_table().uses_pool(pool_)) {
        throw std::invalid_argument("sequence and prefix cache must use one physical pool");
    }
    if (sequence.num_tokens() != 0) {
        throw std::invalid_argument("cached prefix requires an empty sequence");
    }
    const auto block_size = pool_.block_size();
    Tokens prefix;
    prefix.reserve(tokens.size());
    try {
        for (std::size_t offset = 0; tokens.size() - offset >= block_size; offset += block_size) {
            prefix.insert(prefix.end(), tokens.begin() + offset,
                          tokens.begin() + offset + block_size);
            PhysicalBlockHandle handle{};
            if (!pool_.lookup(prefix, handle)) break;
            try {
                // lookup 的临时引用与请求表的长期引用分开；表追加失败时
                // 必须释放前者，已追加的旧块由外层回滚。
                sequence.append_cached_full_block(handle);
            } catch (...) {
                pool_.free(handle);
                throw;
            }
            pool_.free(handle);
        }
    } catch (...) {
        while (sequence.num_tokens_ != 0) {
            sequence.block_table_.pop_back();
            sequence.num_tokens_ -= block_size;
        }
        throw;
    }
    return sequence.num_tokens();
}

void PrefixCache::publish_computed_block(const SequenceState& sequence,
                                         std::size_t logical_block, const Tokens& prefix) {
    if (!sequence.block_table().uses_pool(pool_)) {
        throw std::invalid_argument("sequence and prefix cache must use one physical pool");
    }
    const auto block_size = pool_.block_size();
    if (logical_block >= sequence.num_allocated_blocks() ||
        prefix.size() % block_size != 0 || prefix.size() / block_size != logical_block + 1 ||
        prefix.size() > sequence.num_tokens()) {
        throw std::invalid_argument("publish requires the exact complete token prefix");
    }
    // v0 BlockManager::publish 同时检查唯一持有、重复前缀和哈希碰撞；
    // 发布后零引用页留在缓存队列，容量不足时才可被淘汰并复用编号。
    pool_.publish(sequence.block_handle(logical_block), prefix);
}

} // namespace kvflux::v2

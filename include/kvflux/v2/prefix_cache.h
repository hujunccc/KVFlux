#pragma once

#include "kvflux/v2/sequence_state.h"

namespace kvflux::v2 {

// 将 v0 的前缀哈希、引用计数和零引用缓存队列接到 v2 的请求/物理页映射。
// 本类只管理元数据；与 pool 一样由调用方保证单线程访问。
class PrefixCache {
public:
    explicit PrefixCache(PhysicalBlockPool& pool) noexcept : pool_(pool) {}

    // 在一个空请求上按顺序查找完整前缀块。每次命中为请求保留一个页引用，
    // 返回共享的 token 数；遇到首个 miss 就停止，尾部不足一块永不共享。
    // 异常时撤销本次附加的全部映射。调用方可对剩余 token 调用 append_tokens，
    // 并只计算/写入未命中的 K/V。
    std::size_t attach_cached_prefix(SequenceState& sequence, const Tokens& tokens);

    // 某完整逻辑块的 K/V 已写入并完成同步后，才用真实 token 前缀发布。
    // prefix 必须恰好从请求开头延伸到 logical_block 的末尾；发布只索引
    // 已有物理页，不额外占一个引用。已发布/共享的块不能再次发布。
    void publish_computed_block(const SequenceState& sequence, std::size_t logical_block,
                                const Tokens& prefix);

private:
    PhysicalBlockPool& pool_; // pool 必须比 PrefixCache 和 SequenceState 活得更久。
};

} // namespace kvflux::v2

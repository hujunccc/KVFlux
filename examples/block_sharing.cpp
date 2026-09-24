#include "kvflux/v2/prefix_cache.h"

#include <iostream>

int main() {
    kvflux::PhysicalBlockPool pool(4, 2);
    kvflux::v2::PrefixCache cache(pool);
    {
        kvflux::v2::SequenceState a(1, pool);
        a.append_tokens(5);
        // 此处代表计算后已写完并同步 A 的前两个完整 KV 页。
        cache.publish_computed_block(a, 0, {10, 20});
        cache.publish_computed_block(a, 1, {10, 20, 30, 40});

        kvflux::v2::SequenceState b(2, pool);
        const auto reused = cache.attach_cached_prefix(b, {10, 20, 30, 40, 60});
        b.append_tokens(5 - reused); // B 只为不同的尾 token 申请自己的页。
        std::cout << "reused tokens: " << reused << '\n';
        for (std::size_t logical = 0; logical < 2; ++logical) {
            const auto handle = a.block_table().handle(logical);
            std::cout << "L" << logical << " -> P" << b.physical_block_id(logical)
                      << ", refcount=" << pool.ref_count(handle) << '\n';
        }
        std::cout << "A tail: P" << a.physical_block_id(2)
                  << ", B tail: P" << b.physical_block_id(2) << '\n';
    }
    std::cout << "reclaimable pages after requests: " << pool.available() << '\n';
}

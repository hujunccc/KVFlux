#include "kvflux/v2/prefix_cache.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Error, class Fn> void throws(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

using kvflux::Tokens;
using kvflux::PhysicalBlockPool;
using kvflux::PhysicalBlockHandle;
using kvflux::v2::PrefixCache;
using kvflux::v2::SequenceState;

void shared_prefix_and_private_tail() {
    PhysicalBlockPool pool(6, 2);
    PrefixCache cache(pool);
    kvflux::PhysicalBlockID first, second;
    {
        SequenceState a(1, pool);
        a.append_tokens(5);
        first = a.physical_block_id(0);
        second = a.physical_block_id(1);
        CHECK(first == 0 && second == 1);
        // 模拟完整块 K/V 已写完并同步；未满尾块 P2 不允许发布。
        cache.publish_computed_block(a, 0, {10, 20});
        cache.publish_computed_block(a, 1, {10, 20, 30, 40});
        CHECK(pool.is_published(a.block_table().handle(0)));
        CHECK(!pool.is_published(a.block_table().handle(2)));
        throws<std::invalid_argument>([&] {
            cache.publish_computed_block(a, 2, {10, 20, 30, 40, 50});
        });

        SequenceState b(2, pool);
        CHECK(cache.attach_cached_prefix(b, {10, 20, 30, 40, 60}) == 4);
        CHECK(b.num_tokens() == 4 && b.num_allocated_blocks() == 2);
        CHECK(b.physical_block_id(0) == first && b.physical_block_id(1) == second);
        CHECK(pool.ref_count(a.block_table().handle(0)) == 2);
        CHECK(pool.ref_count(a.block_table().handle(1)) == 2);
        b.append_token();
        CHECK(b.physical_block_id(2) != a.physical_block_id(2));

        SequenceState c(3, pool);
        CHECK(cache.attach_cached_prefix(c, {10, 20, 99}) == 2);
        CHECK(c.physical_block_id(0) == first);
        c.append_token();
        CHECK(pool.ref_count(a.block_table().handle(0)) == 3);

        SequenceState empty(4, pool);
        CHECK(cache.attach_cached_prefix(empty, {7}) == 0);
        CHECK(empty.num_allocated_blocks() == 0);
        throws<std::invalid_argument>([&] {
            cache.attach_cached_prefix(b, {10, 20});
        });
        PhysicalBlockPool other_pool(1, 2);
        SequenceState other(5, other_pool);
        throws<std::invalid_argument>([&] {
            cache.attach_cached_prefix(other, {10, 20});
        });
        throws<std::invalid_argument>([&] {
            cache.publish_computed_block(other, 0, {10, 20});
        });
        CHECK(pool.available() == 1); // P0/P1 共享，P2/P3/P4 为私有尾块。
        const auto spare = pool.allocate();
        throws<kvflux::CapacityError>([&] { (void)pool.allocate(); });
        CHECK(pool.ref_count(a.block_table().handle(0)) == 3);
        pool.free(spare); // 正在共享的 P0/P1 不能被容量压力淘汰。
    }
    // 完整块引用归零后仍可命中；三个私有尾块直接返回 free list。
    CHECK(pool.available() == pool.capacity());
    SequenceState later(6, pool);
    CHECK(cache.attach_cached_prefix(later, {10, 20, 30, 40}) == 4);
    CHECK(later.physical_block_id(0) == first && later.physical_block_id(1) == second);
    CHECK(pool.available() == pool.capacity() - 2);
}

void collision_and_full_prefix_identity() {
    PhysicalBlockPool pool(4, 2, [](const Tokens&) { return 7; });
    PrefixCache cache(pool);
    SequenceState a(1, pool), b(2, pool);
    a.append_tokens(4);
    b.append_tokens(4);
    cache.publish_computed_block(a, 0, {1, 2});
    cache.publish_computed_block(a, 1, {1, 2, 3, 4});
    cache.publish_computed_block(b, 0, {5, 6});
    cache.publish_computed_block(b, 1, {5, 6, 3, 4});

    SequenceState hit(3, pool);
    CHECK(cache.attach_cached_prefix(hit, {5, 6, 3, 4}) == 4);
    CHECK(hit.physical_block_id(0) == b.physical_block_id(0));
    CHECK(hit.physical_block_id(1) == b.physical_block_id(1));
    CHECK(hit.physical_block_id(1) != a.physical_block_id(1));
    SequenceState miss(4, pool);
    CHECK(cache.attach_cached_prefix(miss, {1, 9, 3, 4}) == 0);
    CHECK(miss.num_tokens() == 0);
    throws<std::invalid_argument>([&] {
        cache.publish_computed_block(a, 0, {1, 2}); // 不可二次发布。
    });
}

void idle_cache_is_evicted_only_when_needed() {
    PhysicalBlockPool pool(2, 2);
    PrefixCache cache(pool);
    PhysicalBlockHandle cached{};
    {
        SequenceState a(1, pool);
        a.append_tokens(4);
        cache.publish_computed_block(a, 0, {1, 2});
        cache.publish_computed_block(a, 1, {1, 2, 3, 4});
        CHECK(pool.lookup({1, 2}, cached));
        pool.free(cached);
    }
    CHECK(pool.available() == 2); // 两页均为零引用、可淘汰的缓存。
    const auto reused = pool.allocate(); // 无普通空闲页，淘汰最早闲置的 P0。
    CHECK(reused.id == cached.id && reused.generation != cached.generation);
    PhysicalBlockHandle hit{};
    CHECK(!pool.lookup({1, 2}, hit));
    CHECK(pool.lookup({1, 2, 3, 4}, hit));
    CHECK(hit.id != reused.id && pool.ref_count(hit) == 1);
    pool.free(hit);
    pool.free(reused);
    CHECK(pool.available() == 2);
}

void lookup_failure_rolls_back_attached_refs() {
    bool fail_second_lookup = false;
    PhysicalBlockPool pool(2, 2, [&](const Tokens& prefix) {
        if (fail_second_lookup && prefix.size() == 4) {
            throw std::runtime_error("injected hash failure");
        }
        return kvflux::BlockManager::prefix_hash(prefix);
    });
    PrefixCache cache(pool);
    SequenceState owner(1, pool), newcomer(2, pool);
    owner.append_tokens(4);
    cache.publish_computed_block(owner, 0, {1, 2});
    cache.publish_computed_block(owner, 1, {1, 2, 3, 4});
    fail_second_lookup = true;
    throws<std::runtime_error>([&] {
        cache.attach_cached_prefix(newcomer, {1, 2, 3, 4});
    });
    CHECK(newcomer.num_tokens() == 0 && newcomer.num_allocated_blocks() == 0);
    CHECK(pool.ref_count(owner.block_table().handle(0)) == 1);
    CHECK(pool.ref_count(owner.block_table().handle(1)) == 1);
}

int main() {
    try {
        shared_prefix_and_private_tail();
        collision_and_full_prefix_identity();
        idle_cache_is_evicted_only_when_needed();
        lookup_failure_rolls_back_attached_refs();
        std::cout << "v2 prefix cache: sharing, collision, references and eviction passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

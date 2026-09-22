#include "kvflux/block_manager.h"

#include <iostream>
#include <random>
#include <string>
#include <utility>

using namespace kvflux;

// 不使用 assert：Release 构建定义 NDEBUG 时，测试仍然必须执行检查。
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Error, class Fn> void throws(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

void allocator_and_references() {
    throws<std::invalid_argument>([] { BlockManager m(0, 2); });
    throws<std::invalid_argument>([] { BlockManager m(2, 0); });
    throws<std::invalid_argument>([] { BlockManager m(2, 2, {}); });
    BlockManager m(2, 2);
    auto a = m.allocate(), b = m.allocate();
    CHECK(a.id != b.id);
    throws<CapacityError>([&] { m.allocate(); });
    m.retain(a);
    CHECK(m.ref_count(a) == 2);
    m.release(a);
    CHECK(m.ref_count(a) == 1);
    m.release(a);
    throws<std::invalid_argument>([&] { m.release(a); });
    auto replacement = m.allocate();
    CHECK(replacement.id == a.id);
    CHECK(replacement.generation != a.generation);
    throws<std::invalid_argument>([&] { m.retain(a); });
    throws<std::invalid_argument>([&] { m.release({99, 1}); });
    m.release(replacement);
    m.release(b);
    CHECK(m.stats().free == 2);
}

void prefix_and_lookup() {
    BlockManager m(8, 2);
    auto a = m.acquire({1, 2, 3, 4, 5});
    auto b = m.acquire({1, 2, 3, 4, 5});
    CHECK(a.blocks[0] == b.blocks[0]);
    CHECK(a.blocks[1] == b.blocks[1]);
    CHECK(!(a.blocks[2] == b.blocks[2])); // 相同的不完整尾块也不做前缀缓存。
    auto c = m.acquire({9, 2, 3, 4});
    CHECK(!(a.blocks[1] == c.blocks[1])); // 当前块相同，但之前的上下文不同。
    CHECK(m.block_at(a, 0) == a.blocks[0]);
    CHECK(m.block_at(a, 3) == a.blocks[1]);
    CHECK(m.block_at(a, 4) == a.blocks[2]);
    throws<std::out_of_range>([&] { m.block_at(a, 5); });
    auto shared = m.share(a);
    CHECK(m.ref_count(a.blocks[0]) == 3);
    m.release(shared);
    m.release(a);
    m.release(b);
    m.release(c);
    CHECK(m.stats().active == 0);
    CHECK(m.stats().cached_idle == 4);
    CHECK(m.stats().free == 4);
    m.release(a); // 释放后的空表允许再次释放。
    auto empty = m.acquire({});
    CHECK(empty.blocks.empty());
    m.release(empty);
}

void lru_and_collisions() {
    // 故意让所有前缀哈希相同，同时验证桶内精确查找和淘汰删除。
    BlockManager m(3, 1, [](const Tokens&) { return 7; });
    auto a = m.acquire({1}), b = m.acquire({2}), c = m.acquire({3});
    const auto old_b = b.blocks[0];
    m.release(a);
    m.release(b);
    m.release(c); // LRU 顺序：1 -> 2 -> 3。
    BlockHandle hit{99, 99};
    CHECK(!m.lookup({4}, hit));
    CHECK(hit.id == 99);
    CHECK(m.lookup({1}, hit));
    m.release(hit); // LRU 顺序：2 -> 3 -> 1。
    auto d = m.acquire({4});
    CHECK(d.blocks[0].id == old_b.id);
    CHECK(!m.lookup({2}, hit));
    throws<std::invalid_argument>([&] { m.retain(old_b); });
    CHECK(m.lookup({3}, hit));
    m.release(hit);
    CHECK(m.lookup({1}, hit));
    m.release(hit);
    m.release(d);
    // 活跃块不会淘汰，即使它很早就被访问过。
    CHECK(m.lookup({1}, hit));
    auto e = m.acquire({5}), f = m.acquire({6});
    throws<CapacityError>([&] { m.acquire({7}); });
    CHECK(m.ref_count(hit) == 1);
    m.release(hit);
    m.release(e);
    m.release(f);
}

void rollback_and_validation() {
    BlockManager m(2, 2);
    auto live = m.acquire({1, 2});
    throws<CapacityError>([&] { m.acquire({1, 2, 3, 4, 5}); });
    CHECK(m.ref_count(live.blocks[0]) == 1); // 命中的已有引用也要回滚。
    CHECK(m.stats().active == 1);
    CHECK(m.stats().cached_idle == 1);
    m.release(live);
    auto too_big = [&] { m.acquire({10, 11, 12, 13, 14, 15}); };
    throws<CapacityError>(too_big);
    CHECK(m.stats().active == 0);

    BlockManager raw(3, 2);
    auto h = raw.allocate();
    throws<std::invalid_argument>([&] { raw.publish(h, {}); });
    throws<std::invalid_argument>([&] { raw.publish(h, {1}); });
    raw.retain(h);
    throws<std::invalid_argument>([&] { raw.publish(h, {1, 2}); });
    raw.release(h);
    raw.publish(h, {1, 2});
    throws<std::invalid_argument>([&] { raw.publish(h, {3, 4}); });
    auto other = raw.allocate();
    throws<std::invalid_argument>([&] { raw.publish(other, {1, 2}); });
    BlockTable bad{{h, {999, 1}}, 4};
    throws<std::invalid_argument>([&] { raw.release(bad); });
    CHECK(raw.ref_count(h) == 1);
    BlockTable duplicate{{h, h}, 4};
    throws<std::invalid_argument>([&] { raw.release(duplicate); });
    CHECK(raw.ref_count(h) == 1);
    raw.release(h);
    throws<std::invalid_argument>([&] { raw.release(h); });
    raw.release(other);

    BlockManager throwing(2, 1, [](const Tokens& p) -> std::uint64_t {
        if (p.size() == 2) throw std::runtime_error("simulated hash failure");
        return 0;
    });
    throws<std::runtime_error>([&] { throwing.acquire({1, 2}); });
    CHECK(throwing.stats().active == 0);
    CHECK(BlockManager::prefix_hash({}) == 14695981039346656037ULL);
    CHECK(BlockManager::prefix_hash({-1, 0}) == BlockManager::prefix_hash({-1, 0}));
}

void randomized_reference_model() {
    BlockManager m(12, 2, [](const Tokens&) { return 0; });
    std::mt19937 rng(20260922);
    std::vector<BlockTable> live;
    // 独立统计所有存活表中的引用；每次随机操作后核对管理器的计数。
    for (int step = 0; step < 3000; ++step) {
        const auto action = rng() % 3;
        if (!live.empty() && action == 0) {
            const auto i = rng() % live.size();
            m.release(live[i]);
            live.erase(live.begin() + i);
        } else if (!live.empty() && action == 1 && live.size() < 30) {
            live.push_back(m.share(live[rng() % live.size()]));
        } else {
            Tokens tokens(rng() % 10);
            for (auto& token : tokens) token = static_cast<Token>(rng() % 4);
            try { live.push_back(m.acquire(tokens)); } catch (const CapacityError&) {}
        }
        std::vector<std::size_t> counts(12, 0);
        for (const auto& table : live) for (auto h : table.blocks) ++counts[h.id];
        for (const auto& table : live) for (auto h : table.blocks) CHECK(m.ref_count(h) == counts[h.id]);
        const auto s = m.stats();
        std::size_t active = 0;
        for (auto count : counts) active += count != 0;
        CHECK(s.active == active);
        CHECK(s.free + s.active + s.cached_idle == s.capacity);
    }
    for (auto& table : live) m.release(table);
    CHECK(m.stats().active == 0);
}

int main() {
    try {
        allocator_and_references();
        prefix_and_lookup();
        lru_and_collisions();
        rollback_and_validation();
        randomized_reference_model();
        std::cout << "5 test groups passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

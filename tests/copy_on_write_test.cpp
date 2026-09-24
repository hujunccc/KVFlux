#include "kvflux/v2/sequence_state.h"

#include <array>
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

using kvflux::PhysicalBlockPool;
using kvflux::v2::SequenceState;

void shared_partial_page_copies_before_append() {
    PhysicalBlockPool pool(3, 4);
    std::vector<std::array<int, 4>> pages(pool.capacity()); // 用 CPU 页模拟设备端 K/V。
    {
        SequenceState a(1, pool);
        a.append_tokens(3);
        const auto old = a.physical_block_id(0);
        pages[old] = {11, 22, 33, -1};
        SequenceState b = a.fork_shared(2);
        CHECK(b.num_tokens() == 3 && b.physical_block_id(0) == old);
        CHECK(pool.ref_count(a.block_table().handle(0)) == 2);
        throws<std::logic_error>([&] { a.append_token(); });
        throws<std::logic_error>([&] { a.append_tokens(2); });
        CHECK(a.num_tokens() == 3 && a.physical_block_id(0) == old);

        int copies = 0;
        const auto location = a.append_token_with_copy([&](auto source, auto destination) {
            CHECK(source.id == old && destination.id != old);
            pages[destination.id] = pages[source.id];
            ++copies;
        });
        CHECK(copies == 1 && location.physical_block != old && location.offset_in_block == 3);
        CHECK(a.num_tokens() == 4 && b.num_tokens() == 3);
        CHECK(a.physical_block_id(0) == location.physical_block && b.physical_block_id(0) == old);
        CHECK(pool.ref_count(a.block_table().handle(0)) == 1);
        CHECK(pool.ref_count(b.block_table().handle(0)) == 1);
        CHECK((pages[location.physical_block] == std::array<int, 4>{11, 22, 33, -1}));

        pages[location.physical_block][3] = 44;
        CHECK(pages[old][3] == -1); // A 写新 token 不会改 B 的旧页。
        const auto b_location = b.append_token(); // B 已是旧页的唯一持有者，无需复制。
        CHECK(b_location.physical_block == old && b_location.offset_in_block == 3);
        pages[old][3] = 55;
        CHECK(pages[location.physical_block][3] == 44);
        CHECK(pool.available() == 1);
    }
    CHECK(pool.available() == pool.capacity());
}

void full_page_fork_allocates_a_new_tail() {
    PhysicalBlockPool pool(2, 4);
    SequenceState a(1, pool);
    a.append_tokens(4);
    SequenceState b = a.fork_shared(2);
    bool copied = false;
    const auto location = a.append_token_with_copy([&](auto, auto) { copied = true; });
    CHECK(!copied && location.offset_in_block == 0);
    CHECK(location.physical_block != b.physical_block_id(0));
    CHECK(a.num_tokens() == 5 && b.num_tokens() == 4);
    CHECK(pool.ref_count(b.block_table().handle(0)) == 2);
}

void only_the_shared_tail_is_replaced() {
    PhysicalBlockPool pool(4, 4);
    SequenceState a(1, pool);
    a.append_tokens(11); // L0/L1 已满，L2 只有三个 token。
    SequenceState b = a.fork_shared(2);
    const auto old_tail = b.physical_block_id(2);
    const auto location = a.append_token_with_copy([](auto, auto) {});
    CHECK(location.physical_block != old_tail && location.offset_in_block == 3);
    CHECK(a.physical_block_id(0) == b.physical_block_id(0));
    CHECK(a.physical_block_id(1) == b.physical_block_id(1));
    CHECK(a.physical_block_id(2) != b.physical_block_id(2));
    CHECK(pool.ref_count(a.block_table().handle(0)) == 2);
    CHECK(pool.ref_count(a.block_table().handle(1)) == 2);
    CHECK(pool.ref_count(b.block_table().handle(2)) == 1);
}

void allocation_and_copy_failure_preserve_both_requests() {
    {
        PhysicalBlockPool pool(1, 4);
        SequenceState a(1, pool);
        a.append_tokens(2);
        SequenceState b = a.fork_shared(2);
        throws<kvflux::CapacityError>([&] {
            a.append_token_with_copy([](auto, auto) {});
        });
        CHECK(a.num_tokens() == 2 && b.num_tokens() == 2);
        CHECK(a.physical_block_id(0) == b.physical_block_id(0));
        CHECK(pool.ref_count(a.block_table().handle(0)) == 2);
    }
    {
        PhysicalBlockPool pool(2, 4);
        SequenceState a(1, pool);
        a.append_tokens(2);
        SequenceState b = a.fork_shared(2);
        const auto original = a.physical_block_id(0);
        throws<std::runtime_error>([&] {
            a.append_token_with_copy([](auto, auto) { throw std::runtime_error("copy failed"); });
        });
        CHECK(a.num_tokens() == 2 && b.num_tokens() == 2);
        CHECK(a.physical_block_id(0) == original && b.physical_block_id(0) == original);
        CHECK(pool.ref_count(a.block_table().handle(0)) == 2 && pool.available() == 1);
        throws<std::invalid_argument>([&] { a.append_token_with_copy({}); });
        CHECK(pool.available() == 1);
    }
}

int main() {
    try {
        shared_partial_page_copies_before_append();
        full_page_fork_allocates_a_new_tail();
        only_the_shared_tail_is_replaced();
        allocation_and_copy_failure_preserve_both_requests();
        std::cout << "copy-on-write: shared tail, isolation and rollback passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

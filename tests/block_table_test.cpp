#include "kvflux/v2/block_table.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Error, class Fn> void throws(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

using kvflux::PhysicalBlockHandle;
using kvflux::PhysicalBlockPool;
using kvflux::v2::BlockTable;

static_assert(!std::is_copy_constructible_v<BlockTable>);
static_assert(std::is_move_constructible_v<BlockTable>);

void exact_logical_to_physical_mapping() {
    PhysicalBlockPool pool(100, 16);
    std::vector<PhysicalBlockHandle> initial;
    for (std::size_t i = 0; i < 92; ++i) initial.push_back(pool.allocate());

    BlockTable a(pool);
    for (auto physical : {27U, 91U, 3U, 48U}) a.append_existing(initial[physical]);
    CHECK(a.size() == 4);
    CHECK(a.physical_id(0) == 27);
    CHECK(a[0] == 27);
    CHECK(a.physical_id(1) == 91);
    CHECK(a.physical_id(2) == 3);
    CHECK(a.physical_id(3) == 48);
    throws<std::out_of_range>([&] { a.physical_id(4); });

    // 分配者释放自己的引用后，A 的四个表项仍使这些页保持活跃。
    for (auto handle : initial) pool.free(handle);
    CHECK(pool.available() == 96);

    BlockTable b(pool);
    b.append_shared(a, 1);
    b.append_shared(a, 3);
    CHECK(b.physical_id(0) == 91 && b.physical_id(1) == 48);
    CHECK(pool.available() == 96); // 共享只增加引用，不占新物理页。

    a.pop_back(); // A 归还 48，但 B 仍持有它。
    CHECK(a.size() == 3 && b.physical_id(1) == 48);
    a.clear();
    CHECK(a.empty() && pool.available() == 98);
    CHECK(b.physical_id(0) == 91);
    b.clear();
    CHECK(pool.available() == pool.capacity());
    throws<std::out_of_range>([&] { b.pop_back(); });
}

void capacity_and_move_lifecycle() {
    PhysicalBlockPool pool(2, 16);
    BlockTable a(pool);
    CHECK(a.append_new() == 0);
    CHECK(a.append_new() == 1);
    throws<kvflux::CapacityError>([&] { a.append_new(); });
    CHECK(a.size() == 2 && pool.available() == 0);

    // 表的移动转移引用所有权，源表析构不能再次释放这些页。
    BlockTable moved(std::move(a));
    CHECK(a.empty() && moved.physical_id(0) == 0);
    throws<std::logic_error>([&] { a.append_new(); });
    moved.append_shared(moved, 0); // 同一编号在表中出现两次，各持有一个引用。
    CHECK(moved.size() == 3 && pool.available() == 0);

    BlockTable destination(pool);
    destination = std::move(moved);
    CHECK(moved.empty() && destination.size() == 3);
    destination.pop_back();
    CHECK(pool.available() == 0); // 第 0 页仍被表中另一项引用。
    destination.clear();
    CHECK(pool.available() == 2);
}

void invalid_inputs_do_not_change_table() {
    PhysicalBlockPool pool(2, 16);
    PhysicalBlockPool other_pool(2, 16);
    BlockTable a(pool), other(other_pool);
    auto stale = pool.allocate();
    pool.free(stale);
    throws<std::invalid_argument>([&] { a.append_existing(stale); });
    CHECK(a.empty() && pool.available() == 2);

    a.append_new();
    throws<std::out_of_range>([&] { a.append_shared(a, 1); });
    throws<std::invalid_argument>([&] { a.append_shared(other, 0); });
    CHECK(a.size() == 1 && pool.available() == 1);
}

void destructor_and_move_assignment_release_refs() {
    PhysicalBlockPool first_pool(2, 16), second_pool(2, 16);
    {
        BlockTable first(first_pool);
        first.append_new();
        BlockTable second(second_pool);
        second.append_new();
        CHECK(first_pool.available() == 1 && second_pool.available() == 1);
        second = std::move(first); // second 原来的引用应立即归还。
        CHECK(second_pool.available() == 2);
        CHECK(first_pool.available() == 1 && second[0] == 0);
    } // second 析构归还移动过来的引用；first 已变空。
    CHECK(first_pool.available() == 2 && second_pool.available() == 2);
}

int main() {
    try {
        exact_logical_to_physical_mapping();
        capacity_and_move_lifecycle();
        invalid_inputs_do_not_change_table();
        destructor_and_move_assignment_release_refs();
        std::cout << "v2 block table: mapping and ownership checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

#include "kvflux/v2/sequence_state.h"

#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Fn> void capacity_error(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const kvflux::CapacityError&) { caught = true; }
    CHECK(caught);
}

void multi_page_allocation_rolls_back() {
    kvflux::PhysicalBlockPool pool(2, 4);
    {
        kvflux::v2::SequenceState request(1, pool);
        request.append_tokens(4);
        const auto original = request.block_table().handle(0);

        // 此次 append 要两页，但池只剩一页。第一笔分配也必须被撤销。
        capacity_error([&] { request.append_tokens(8); });
        CHECK(request.num_tokens() == 4 && request.num_allocated_blocks() == 1);
        CHECK(request.block_table().handle(0) == original);
        CHECK(pool.ref_count(original) == 1 && pool.available() == 1);

        kvflux::v2::SequenceState blocker(2, pool);
        blocker.append_token();
        capacity_error([&] { request.append_token(); });
        CHECK(request.num_tokens() == 4 && request.block_table().handle(0) == original);
        CHECK(pool.available() == 0);
    }
    CHECK(pool.available() == pool.capacity());
}

void shared_page_stays_allocated_until_last_owner_exits() {
    kvflux::PhysicalBlockPool pool(1, 4);
    {
        kvflux::v2::SequenceState a(1, pool);
        a.append_tokens(2);
        const auto original = a.block_table().handle(0);
        {
            auto b = a.fork_shared(2);
            CHECK(pool.ref_count(original) == 2 && pool.available() == 0);
            capacity_error([&] { a.append_token_with_copy([](auto, auto) {}); });
            CHECK(a.num_tokens() == 2 && b.num_tokens() == 2);
            CHECK(a.block_table().handle(0) == original);
            CHECK(b.block_table().handle(0) == original);
            CHECK(pool.ref_count(original) == 2);
        }
        CHECK(pool.ref_count(original) == 1 && pool.available() == 0);
        a.append_token(); // B 退出后，A 可直接写入仍由它持有的尾页。
        CHECK(a.num_tokens() == 3 && pool.available() == 0);
        capacity_error([&] { a.append_tokens(2); }); // 第五个 token 才需要新页。
        CHECK(a.num_tokens() == 3 && a.block_table().handle(0) == original);
    }
    CHECK(pool.available() == 1);
    auto reused = pool.allocate();
    CHECK(pool.id(reused) == 0);
    pool.free(reused);
}

int main() {
    try {
        multi_page_allocation_rolls_back();
        shared_page_stays_allocated_until_last_owner_exits();
        std::cout << "OOM: CapacityError preserves tables, refs and reuse\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

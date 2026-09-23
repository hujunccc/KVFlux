#include "kvflux/v2/sequence_state.h"

#include <iostream>
#include <limits>
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
using kvflux::v2::SequenceState;

static_assert(!std::is_copy_constructible_v<SequenceState>);
static_assert(std::is_move_constructible_v<SequenceState>);

void request_1001_has_virtual_blocks() {
    PhysicalBlockPool pool(100, 16);
    std::vector<PhysicalBlockHandle> occupied;
    auto occupy_until = [&](std::size_t id) {
        while (occupied.size() < id) occupied.push_back(pool.allocate());
    };

    SequenceState request(1001, pool);
    CHECK(request.request_id() == 1001 && request.block_size() == 16);
    CHECK(request.num_tokens() == 0 && request.num_allocated_blocks() == 0);
    CHECK(request.last_block_num_tokens() == 0);

    occupy_until(7);
    request.append_tokens(16); // logical 0 -> physical 7
    occupy_until(32);
    request.append_tokens(16); // logical 1 -> physical 33
    occupy_until(89);
    request.append_tokens(3);  // logical 2 -> physical 91

    CHECK(request.num_tokens() == 35);
    CHECK(request.num_allocated_blocks() == 3);
    CHECK(request.last_block_num_tokens() == 3);
    CHECK(request.block_table().size() == 3);
    CHECK(request.physical_block_id(0) == 7);
    CHECK(request.physical_block_id(1) == 33);
    CHECK(request.physical_block_id(2) == 91);
    CHECK(request.token_location(0).physical_block == 7);
    CHECK(request.token_location(16).physical_block == 33);
    CHECK(request.token_location(34).physical_block == 91);
    CHECK(request.token_location(34).offset_in_block == 2);
    throws<std::out_of_range>([&] { request.token_location(35); });
    throws<std::out_of_range>([&] { request.physical_block_id(3); });

    for (auto handle : occupied) pool.free(handle);
    CHECK(pool.available() == 97); // 请求仍拥有 7、33、91。
}

void tail_growth_and_failure_rollback() {
    PhysicalBlockPool pool(2, 16);
    SequenceState request(42, pool);
    request.append_tokens(15);
    CHECK(request.num_allocated_blocks() == 1 && request.last_block_num_tokens() == 15);
    request.append_tokens(1);
    CHECK(request.num_allocated_blocks() == 1 && request.last_block_num_tokens() == 16);
    request.append_tokens(0);
    CHECK(request.num_tokens() == 16 && pool.available() == 1);

    // 需要两页，但池只有一页：已申请的新页应回滚。
    throws<kvflux::CapacityError>([&] { request.append_tokens(32); });
    CHECK(request.num_tokens() == 16 && request.num_allocated_blocks() == 1);
    CHECK(request.last_block_num_tokens() == 16 && pool.available() == 1);
    throws<std::overflow_error>([&] {
        request.append_tokens(std::numeric_limits<std::size_t>::max());
    });
    CHECK(request.num_tokens() == 16 && pool.available() == 1);

    request.append_tokens(1);
    CHECK(request.num_tokens() == 17 && request.num_allocated_blocks() == 2);
    CHECK(request.last_block_num_tokens() == 1 && pool.available() == 0);
}

void move_transfers_request_ownership() {
    PhysicalBlockPool first_pool(2, 16), second_pool(1, 8);
    {
        SequenceState first(1001, first_pool);
        first.append_tokens(17);
        SequenceState second(2002, second_pool);
        second.append_tokens(1);
        second = std::move(first);
        CHECK(second_pool.available() == 1);
        CHECK(first.num_tokens() == 0 && first.num_allocated_blocks() == 0);
        CHECK(second.request_id() == 1001 && second.num_tokens() == 17);
        CHECK(second.block_size() == 16 && second.num_allocated_blocks() == 2);
        CHECK(first_pool.available() == 0);
    }
    CHECK(first_pool.available() == 2 && second_pool.available() == 1);
}

int main() {
    try {
        request_1001_has_virtual_blocks();
        tail_growth_and_failure_rollback();
        move_transfers_request_ownership();
        std::cout << "v2 sequence state: virtual mapping and lifecycle checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

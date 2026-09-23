#include "kvflux/v2/slot_mapping.h"

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

using kvflux::PhysicalBlockHandle;
using kvflux::PhysicalBlockPool;
using kvflux::v2::SequenceState;
using kvflux::v2::TokenWrite;
using kvflux::v2::build_slot_mapping;

void ordered_decode_batch_uses_physical_slots() {
    PhysicalBlockPool pool(160, 16);
    std::vector<PhysicalBlockHandle> occupied;
    auto occupy_until_next_id = [&](std::size_t next_id) {
        while (pool.capacity() - pool.available() < next_id) {
            occupied.push_back(pool.allocate());
        }
    };

    {
        occupy_until_next_id(31);
        SequenceState b(1002, pool);
        b.append_tokens(18); // P31、P32；token 17 在 P32 的偏移 1。

        occupy_until_next_id(79);
        SequenceState a(1001, pool);
        a.append_tokens(35); // P79、P80、P81；token 34 在 P81 的偏移 2。

        occupy_until_next_id(133);
        SequenceState c(1003, pool);
        c.append_tokens(81); // P133..P138；token 80 在 P138 的偏移 0。

        const std::vector<TokenWrite> batch{{&a, 34}, {&b, 17}, {&c, 80}};
        const auto slots = build_slot_mapping(batch);
        CHECK(slots.size() == 3);
        CHECK(slots[0] == 1298); // 81 * 16 + 2
        CHECK(slots[1] == 513);  // 32 * 16 + 1
        CHECK(slots[2] == 2208); // 138 * 16 + 0
        CHECK(a.num_tokens() == 35 && b.num_tokens() == 18 && c.num_tokens() == 81);

        c.append_tokens(8);
        const auto later = build_slot_mapping({{&c, 88}});
        CHECK(later.size() == 1 && later[0] == 2216); // P138 的偏移 8。
    }
    for (auto handle : occupied) pool.free(handle);
    CHECK(pool.available() == pool.capacity());
}

void invalid_batch_is_rejected_without_mutation() {
    PhysicalBlockPool first_pool(2, 16), second_pool(2, 16);
    SequenceState first(1, first_pool), second(2, second_pool);
    first.append_tokens(16);
    second.append_tokens(1);

    CHECK(build_slot_mapping({}).empty());
    const auto same_request = build_slot_mapping({{&first, 15}, {&first, 0}});
    CHECK(same_request.size() == 2 && same_request[0] == 15 && same_request[1] == 0);
    throws<std::invalid_argument>([&] { build_slot_mapping({{nullptr, 0}}); });
    throws<std::invalid_argument>([&] { build_slot_mapping({{&first, 0}, {nullptr, 0}}); });
    throws<std::out_of_range>([&] { build_slot_mapping({{&first, 16}}); });
    throws<std::invalid_argument>([&] { build_slot_mapping({{&first, 0}, {&second, 0}}); });
    CHECK(first.num_tokens() == 16 && second.num_tokens() == 1);
    CHECK(first_pool.available() == 1 && second_pool.available() == 1);
}

int main() {
    try {
        ordered_decode_batch_uses_physical_slots();
        invalid_batch_is_rejected_without_mutation();
        std::cout << "slot mapping: ordered batch and invalid-input checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

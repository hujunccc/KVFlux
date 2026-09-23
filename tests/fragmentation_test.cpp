#include "kvflux/v2/sequence_state.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

int main() {
    try {
        constexpr std::size_t capacity = 8;
        constexpr std::size_t block_size = 16;
        kvflux::PhysicalBlockPool pool(capacity, block_size);
        std::array<kvflux::PhysicalBlockHandle, capacity> holders{};

        for (std::size_t id = 0; id < capacity; ++id) {
            holders[id] = pool.allocate();
            CHECK(pool.id(holders[id]) == id);
        }
        CHECK(pool.available() == 0);

        // 释放奇数页；反向归还使 LIFO free list 的分配顺序为 1、3、5、7。
        for (std::size_t next = capacity; next > 0; next -= 2) pool.free(holders[next - 1]);
        CHECK(pool.available() == 4);

        std::size_t longest_free_run = 0;
        std::size_t current_free_run = 0;
        for (std::size_t id = 0; id < capacity; ++id) {
            if (id % 2 == 0) {
                CHECK(pool.id(holders[id]) == id && pool.ref_count(holders[id]) == 1);
                current_free_run = 0;
            } else {
                bool rejected_stale_handle = false;
                try { pool.id(holders[id]); }
                catch (const std::invalid_argument&) { rejected_stale_handle = true; }
                CHECK(rejected_stale_handle);
                ++current_free_run;
                if (current_free_run > longest_free_run) longest_free_run = current_free_run;
            }
        }
        CHECK(longest_free_run == 1); // 有 4 页空闲，却没有连续的 4 页。
        std::cout << "Initial free pages: P1 P3 P5 P7; longest contiguous run: "
                  << longest_free_run << '\n';

        {
            kvflux::v2::SequenceState request_x(1001, pool);
            request_x.append_tokens(4 * block_size);
            CHECK(request_x.num_allocated_blocks() == 4 && pool.available() == 0);
            CHECK(request_x.num_tokens() == 64 && request_x.last_block_num_tokens() == 16);
            for (std::size_t logical = 0; logical < 4; ++logical) {
                const auto expected = 2 * logical + 1;
                CHECK(request_x.block_table()[logical] == expected);
                CHECK(request_x.token_location(logical * block_size).physical_block == expected);
                CHECK(request_x.token_location(logical * block_size).offset_in_block == 0);
            }
            CHECK(request_x.token_location(63).physical_block == 7);
            CHECK(request_x.token_location(63).offset_in_block == 15);
            for (std::size_t id = 0; id < capacity; id += 2) {
                CHECK(pool.id(holders[id]) == id && pool.ref_count(holders[id]) == 1);
            }
            std::cout << "Request X block table: [1, 3, 5, 7]\n";
        }

        CHECK(pool.available() == 4); // 请求释放了四个非连续页。
        for (std::size_t id = 0; id < capacity; id += 2) pool.free(holders[id]);
        CHECK(pool.available() == capacity);
        std::cout << "Fragmentation test passed; all 8 pages returned\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

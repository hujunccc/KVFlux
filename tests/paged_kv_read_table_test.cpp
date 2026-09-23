#include "kvflux/v2/paged_kv_read.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

int main() {
    try {
        kvflux::PhysicalBlockPool pool(96, 16);
        std::vector<kvflux::PhysicalBlockHandle> occupied;
        for (std::size_t i = 0; i < 27; ++i) occupied.push_back(pool.allocate());
        {
            kvflux::v2::SequenceState request(1001, pool);
            CHECK(kvflux::v2::build_read_block_table(request).empty());
            request.append_tokens(16); // P27
            for (std::size_t i = 28; i < 91; ++i) occupied.push_back(pool.allocate());
            pool.free(occupied[3]); // P3 回到 free list 头部。
            request.append_tokens(16); // P3
            request.append_tokens(8);  // P91
            CHECK(request.num_tokens() == 40);
            CHECK(request.last_block_num_tokens() == 8);
            CHECK((kvflux::v2::build_read_block_table(request) ==
                   kvflux::v2::ReadBlockTable{27, 3, 91}));
            CHECK(request.token_location(0).physical_block == 27);
            CHECK(request.token_location(16).physical_block == 3);
            CHECK(request.token_location(32).physical_block == 91);
        }
        for (std::size_t i = 0; i < occupied.size(); ++i) {
            if (i != 3) pool.free(occupied[i]);
        }
        CHECK(pool.available() == pool.capacity());
        std::cout << "paged KV read table: [27, 3, 91] for 40 tokens\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

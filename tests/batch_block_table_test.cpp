#include "kvflux/v2/batch_block_table.h"

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

int main() {
    try {
        using kvflux::v2::BatchBlockTable;
        using kvflux::v2::SequenceState;
        using kvflux::v2::build_batch_block_table;
        CHECK(build_batch_block_table({}).num_sequences() == 0);

        kvflux::PhysicalBlockPool pool(16, 16);
        {
            SequenceState a(1, pool), b(2, pool), c(3, pool), empty(4, pool);
            // 交错增长使每行的物理编号不连续：A=[0,3,6]、B=[1,4,7,8]、C=[2,5]。
            a.append_tokens(16);
            b.append_tokens(16);
            c.append_tokens(16);
            a.append_tokens(16);
            b.append_tokens(16);
            c.append_tokens(12);
            a.append_tokens(3);
            b.append_tokens(29);
            const auto batch = build_batch_block_table({&a, &b, &c, &empty});
            const auto pad = BatchBlockTable::invalid_block();
            CHECK(batch.num_sequences() == 4);
            CHECK(batch.max_blocks_per_sequence() == 4);
            CHECK((batch.sequence_lengths() == std::vector<std::size_t>{35, 61, 28, 0}));
            CHECK((batch.block_table() == std::vector<kvflux::PhysicalBlockID>{
                0, 3, 6, pad, 1, 4, 7, 8, 2, 5, pad, pad, pad, pad, pad, pad}));
            CHECK(batch.physical_id(1, 3) == 8);
            CHECK(batch.physical_id(2, 1) == 5);
            throws<std::out_of_range>([&] { (void)batch.physical_id(0, 3); });
            throws<std::out_of_range>([&] { (void)batch.physical_id(3, 0); });
            throws<std::out_of_range>([&] { (void)batch.physical_id(4, 0); });
            throws<std::invalid_argument>([&] { (void)build_batch_block_table({&a, nullptr}); });
            kvflux::PhysicalBlockPool other_pool(2, 16);
            SequenceState other(5, other_pool);
            throws<std::invalid_argument>([&] { (void)build_batch_block_table({&a, &other}); });
        }
        CHECK(pool.available() == pool.capacity());
        std::cout << "batch block table: [35,61,28,0] with padded rows passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

#include "kvflux/v2/slot_mapping.h"

#include <iostream>
#include <vector>

int main() {
    kvflux::PhysicalBlockPool pool(160, 16);
    std::vector<kvflux::PhysicalBlockHandle> occupied;
    auto occupy_until_next_id = [&](std::size_t next_id) {
        while (pool.capacity() - pool.available() < next_id) {
            occupied.push_back(pool.allocate());
        }
    };

    {
        // 其他请求占据间隔页，模拟三个请求的不连续 Block Table。
        occupy_until_next_id(31);
        kvflux::v2::SequenceState b(1002, pool);
        b.append_tokens(18);
        occupy_until_next_id(79);
        kvflux::v2::SequenceState a(1001, pool);
        a.append_tokens(35);
        occupy_until_next_id(133);
        kvflux::v2::SequenceState c(1003, pool);
        c.append_tokens(81);

        const std::vector<kvflux::v2::TokenWrite> batch{{&a, 34}, {&b, 17}, {&c, 80}};
        const auto slot_mapping = kvflux::v2::build_slot_mapping(batch);
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const auto location = batch[i].sequence->token_location(batch[i].token_position);
            std::cout << "Request " << batch[i].sequence->request_id()
                      << " token " << batch[i].token_position
                      << " -> P" << location.physical_block
                      << " offset " << location.offset_in_block
                      << " -> slot " << slot_mapping[i] << '\n';
        }
    }
    for (auto handle : occupied) pool.free(handle);
    std::cout << "Available blocks after release: " << pool.available() << '\n';
}

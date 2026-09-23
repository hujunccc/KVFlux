#include "kvflux/v2/sequence_state.h"

#include <iostream>
#include <vector>

int main() {
    kvflux::PhysicalBlockPool pool(100, 16);
    std::vector<kvflux::PhysicalBlockHandle> occupied;
    auto occupy_until = [&](std::size_t count) {
        while (occupied.size() < count) occupied.push_back(pool.allocate());
    };

    {
        // 其他请求占用物理页，让本请求的两页故意不连续。
        occupy_until(37);
        kvflux::v2::SequenceState request(1001, pool);
        request.append_tokens(15);
        std::cout << "15 tokens: [P" << request.block_table()[0] << "]\n";

        const auto sixteenth = request.append_token();
        std::cout << "16 tokens: [P" << request.block_table()[0]
                  << "], new token at offset " << sixteenth.offset_in_block << '\n';

        occupy_until(90);
        const auto seventeenth = request.append_token();
        std::cout << "17 tokens: [P" << request.block_table()[0]
                  << ", P" << request.block_table()[1]
                  << "], new token at offset " << seventeenth.offset_in_block << '\n';
    }
    for (auto handle : occupied) pool.free(handle);
    std::cout << "Available blocks after release: " << pool.available() << '\n';
}

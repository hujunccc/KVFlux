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
        kvflux::v2::SequenceState request(1001, pool);
        // 占用其他物理页，演示逻辑编号与物理编号不必连续或相同。
        occupy_until(7);
        request.append_tokens(16);
        occupy_until(32);
        request.append_tokens(16);
        occupy_until(89);
        request.append_tokens(3);

        std::cout << "Request " << request.request_id()
                  << ": num_tokens=" << request.num_tokens()
                  << ", num_allocated_blocks=" << request.num_allocated_blocks()
                  << ", last_block_num_tokens=" << request.last_block_num_tokens() << '\n';
        for (std::size_t logical = 0; logical < request.num_allocated_blocks(); ++logical) {
            std::cout << "Logical " << logical << " -> P"
                      << request.physical_block_id(logical) << '\n';
        }
        const auto last = request.token_location(34);
        std::cout << "Token 34 -> P" << last.physical_block
                  << ", offset " << last.offset_in_block << '\n';
    }
    for (auto handle : occupied) pool.free(handle);
    std::cout << "Available physical blocks after request: " << pool.available() << '\n';
}

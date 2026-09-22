#include "kvflux/block_manager.h"
#include <iostream>

int main() {
    // 4 个物理槽位，每块容纳 2 个 token；此例不存储真实 K/V 张量。
    kvflux::BlockManager manager(4, 2);
    auto first = manager.acquire({10, 20, 30, 40, 50});
    auto second = manager.acquire({10, 20, 30, 40, 60});

    std::cout << "shared prefix blocks: "
              << (first.blocks[0] == second.blocks[0]) +
                     (first.blocks[1] == second.blocks[1]) << '\n';
    std::cout << "token 4 -> physical block "
              << manager.block_at(second, 4).id << ", offset "
              << 4 % manager.block_size() << '\n';

    // 两个请求均结束后，完整块留在缓存，两个不完整尾块回到 free list。
    manager.release(first);
    manager.release(second);
    const auto stats = manager.stats();
    std::cout << "free=" << stats.free << " active=" << stats.active
              << " cached_idle=" << stats.cached_idle << '\n';
}

#include "kvflux/v2/block_table.h"

#include <iostream>
#include <vector>

int main() {
    kvflux::PhysicalBlockPool pool(100, 16);
    std::vector<kvflux::PhysicalBlockHandle> pages;
    for (std::size_t i = 0; i < 92; ++i) pages.push_back(pool.allocate());

    kvflux::v2::BlockTable sequence_a(pool);
    // 表的下标就是 logical block ID；这里故意选不连续的物理编号。
    for (auto id : {27U, 91U, 3U, 48U}) sequence_a.append_existing(pages[id]);
    for (auto handle : pages) pool.free(handle); // 表已经为四个映射各持有一个引用。

    for (std::size_t logical = 0; logical < sequence_a.size(); ++logical) {
        std::cout << "logical block " << logical << " -> physical block "
                  << sequence_a[logical] << '\n';
    }
    sequence_a.clear();
    std::cout << "available physical blocks: " << pool.available() << '\n';
}

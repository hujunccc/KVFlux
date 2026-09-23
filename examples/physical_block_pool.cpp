#include "kvflux/physical_block_pool.h"

#include <iostream>

int main() {
    kvflux::PhysicalBlockPool pool(1024, 16); // 1024 个物理页，每页容纳 16 个 token。

    const auto a = pool.allocate();
    const auto b = pool.allocate();
    const auto c = pool.allocate();
    std::cout << "allocated IDs: " << pool.id(a) << ' ' << pool.id(b) << ' '
              << pool.id(c) << '\n';

    pool.free(b);
    const auto d = pool.allocate();
    std::cout << "reused ID: " << pool.id(d)
              << " (generation " << b.generation << " -> " << d.generation << ")\n";

    pool.free(a);
    pool.free(c);
    pool.free(d);
    std::cout << "available: " << pool.available() << '/' << pool.capacity() << '\n';
}

#include "kvflux/tiered_block_manager.h"
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    try {
        kvflux::TieredBlockManager manager(8, 12, 4096, 16);
        std::vector<kvflux::LogicalBlockHandle> blocks;
        std::vector<unsigned char> data(4096), result(4096);
        for (int i = 0; i < 12; ++i) {
            std::fill(data.begin(), data.end(), static_cast<unsigned char>(i));
            blocks.push_back(manager.create(data.data(), data.size()));
        }
        auto s = manager.stats();
        std::cout << "12 logical blocks: GPU=" << s.gpu_resident << " CPU=" << s.cpu_resident
                  << " offloads=" << s.offloads << '\n';
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            {
                auto lease = manager.acquire_gpu(blocks[i]);
                manager.prefetch_next(blocks, i, 2);
                // 真正的 kernel 可在这里使用 lease.data()；结束后才能销毁 lease。
                manager.poll();
            }
            manager.read_block(blocks[i], result.data(), result.size());
            for (auto value : result) if (value != i) throw std::runtime_error("tiered data mismatch");
        }
        manager.wait();
        std::cout << manager.metrics();
        for (auto h : blocks) manager.release(h);
        std::cout << "roundtrip=exact, GPU free=" << manager.stats().gpu_free << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

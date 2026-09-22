#include "kvflux/gpu_memory_pool.h"
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    try {
        kvflux::KVBlockLayout layout({2, 4, 8, 16, kvflux::DType::Float32});
        // 池在服务启动时建立。这里用小配置，只申请 20 个 block。
        kvflux::GpuMemoryPool pool(layout, 20 * layout.block_bytes());
        std::vector<kvflux::BlockHandle> blocks;
        for (int i = 0; i < 18; ++i) blocks.push_back(pool.allocate());
        const auto h = blocks[17];
        std::vector<float> input(layout.block_bytes() / sizeof(float)), output(input.size());
        std::iota(input.begin(), input.end(), 0.0f);
        pool.write_block(h, input.data(), layout.block_bytes());
        pool.read_block(h, output.data(), layout.block_bytes());
        if (input != output) throw std::runtime_error("GPU roundtrip mismatch");
        std::cout << "BlockId=" << h.id << " GPU address=" << pool.device_address(h)
                  << " block_bytes=" << pool.block_bytes() << "\nroundtrip=exact"
                  << " total=" << pool.total_blocks() << " remaining=" << pool.remaining_blocks() << '\n';
        for (auto block : blocks) pool.release(block);
        std::cout << "after release: remaining=" << pool.remaining_blocks() << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

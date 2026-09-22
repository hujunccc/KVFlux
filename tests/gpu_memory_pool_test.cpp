#include "kvflux/gpu_memory_pool.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#define CHECK(c) do { if (!(c)) throw std::runtime_error("check failed: " #c); } while (false)
template<class Error, class F> void throws(F fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}
using namespace kvflux;

void roundtrip(DType dtype, std::size_t count) {
    KVBlockLayout layout({2, 3, 5, 4, dtype});
    GpuMemoryPool pool(layout, count * layout.block_bytes() + 1);
    CHECK(pool.total_blocks() == count && pool.remaining_blocks() == count);
    CHECK(pool.allocated_bytes() == count * layout.block_bytes());
    std::vector<BlockHandle> handles;
    std::vector<std::vector<unsigned char>> expected(count, std::vector<unsigned char>(layout.block_bytes()));
    std::mt19937 random(1234);
    auto* base = static_cast<unsigned char*>(pool.device_address(BlockId{0}));
    cudaPointerAttributes attributes{};
    CHECK(cudaPointerGetAttributes(&attributes, base) == cudaSuccess);
    CHECK(attributes.type == cudaMemoryTypeDevice); // 确认真的是显存，不能用 host mock 代替。
    CHECK(attributes.device == pool.device());
    for (std::size_t id = 0; id < count; ++id) {
        auto h = pool.allocate();
        CHECK(h.id == id);
        CHECK(pool.device_address(h) == base + id * layout.block_bytes());
        CHECK(pool.remaining_blocks() == count - id - 1);
        handles.push_back(h);
    }
    throws<CapacityError>([&] { pool.allocate(); });
    std::vector<unsigned char> actual(layout.block_bytes());
    throws<std::invalid_argument>([&] { pool.read_block(handles[0], actual.data(), actual.size()); });
    throws<std::invalid_argument>([&] { pool.publish(handles[0], {1, 2, 3, 4}); });
    throws<std::out_of_range>([&] { pool.device_address(count); });
    throws<std::invalid_argument>([&] { pool.write_block(handles[0], nullptr, actual.size()); });
    throws<std::invalid_argument>([&] { pool.write_block(handles[0], actual.data(), actual.size() - 1); });
    throws<std::invalid_argument>([&] { pool.read_block(handles[0], nullptr, actual.size()); });
    throws<std::invalid_argument>([&] { pool.read_block(handles[0], actual.data(), actual.size() + 1); });

    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    for (int round = 0; round < 8; ++round) {
        // 连续覆盖写多个不同 id，再随机顺序读回，检测相邻块越界和写串块。
        for (std::size_t id = 0; id < count; ++id) {
            for (auto& byte : expected[id]) byte = static_cast<unsigned char>(random());
            pool.write_block(handles[id], expected[id].data(), expected[id].size());
        }
        std::shuffle(order.begin(), order.end(), random);
        for (auto id : order) {
            pool.read_block(handles[id], actual.data(), actual.size());
            CHECK(actual == expected[id]); // 按字节精确相等，包括 FP16/BF16 原始位模式。
        }
        CHECK(pool.device_address(BlockId{0}) == base); // 请求期间池地址保持不变。
    }
    pool.retain(handles[0]);
    throws<std::invalid_argument>([&] { pool.write_block(handles[0], actual.data(), actual.size()); });
    pool.release(handles[0]);
    for (auto h : handles) pool.release(h);
    CHECK(pool.remaining_blocks() == count);
    auto replacement = pool.allocate();
    CHECK(replacement.id == handles.back().id);
    CHECK(replacement.generation != handles.back().generation);
    throws<std::invalid_argument>([&] { pool.device_address(handles.back()); });
    throws<std::invalid_argument>([&] { pool.read_block(replacement, actual.data(), actual.size()); });
    pool.release(replacement);
}

void cache_lifecycle() {
    GpuMemoryPool pool(1, 64, 2);
    std::vector<unsigned char> input(64, 17), output(64);
    auto h = pool.allocate();
    pool.write_block(h, input.data(), input.size());
    pool.publish(h, {1, 2});
    throws<std::invalid_argument>([&] { pool.write_block(h, input.data(), input.size()); });
    pool.release(h);
    CHECK(pool.stats().free == 0 && pool.stats().cached_idle == 1);
    CHECK(pool.remaining_blocks() == 1);
    BlockHandle hit{};
    CHECK(pool.lookup({1, 2}, hit));
    CHECK(hit == h && pool.remaining_blocks() == 0);
    pool.read_block(hit, output.data(), output.size());
    CHECK(input == output);
    pool.release(hit);
    auto next = pool.allocate(); // LRU 淘汰仅复用槽位，显存地址不变。
    CHECK(next.id == h.id && next.generation != h.generation);
    CHECK(!pool.lookup({1, 2}, hit));
    throws<std::invalid_argument>([&] { pool.publish(next, {3, 4}); });
    input.assign(64, 99);
    pool.write_block(next, input.data(), input.size());
    pool.read_block(next, output.data(), output.size());
    CHECK(input == output);
    pool.release(next);
}

int main() {
    try {
        int count = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        int previous = -1;
        CHECK(cudaGetDevice(&previous) == cudaSuccess);
        for (auto dtype : {DType::Float16, DType::BFloat16, DType::Float32}) {
            roundtrip(dtype, 1);
            roundtrip(dtype, 20); // 包含用户示例中的 block #17。
        }
        cache_lifecycle();
        throws<std::invalid_argument>([] { GpuMemoryPool p(0, 64, 2); });
        throws<std::invalid_argument>([] { GpuMemoryPool p(2, 0, 2); });
        throws<std::invalid_argument>([] { GpuMemoryPool p(2, 64, 0); });
        throws<std::invalid_argument>([] { GpuMemoryPool p(2, 64, 2, -1); });
        throws<std::overflow_error>([] { GpuMemoryPool p(2, std::numeric_limits<std::size_t>::max(), 2); });
        throws<std::invalid_argument>([] {
            KVBlockLayout layout({1, 1, 1, 1});
            GpuMemoryPool p(layout, layout.block_bytes() - 1);
        });
        throws<std::invalid_argument>([&] { GpuMemoryPool p(1, 64, 2, count); });
        if (count > 1) {
            // 多卡机器上额外检查池操作没有更改调用方的当前设备。
            CHECK(cudaSetDevice(1) == cudaSuccess);
            { GpuMemoryPool p(1, 64, 2, 0); }
            int current = -1;
            CHECK(cudaGetDevice(&current) == cudaSuccess && current == 1);
            CHECK(cudaSetDevice(previous) == cudaSuccess);
        }
        std::cout << "GPU pool: all byte-exact roundtrip and lifecycle checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

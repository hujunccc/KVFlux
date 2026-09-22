#include "kvflux/async_transfer.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#define CHECK(c) do { if (!(c)) throw std::runtime_error("check failed: " #c); } while (false)
template<class Error, class F> void throws(F fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}
using namespace kvflux;

void batch_roundtrip(std::size_t n) {
    constexpr std::size_t bytes = 4096;
    GpuMemoryPool pool(n, bytes, 16);
    AsyncTransferRuntime runtime(pool);
    CHECK(runtime.compute_stream().native_handle() != runtime.transfer_stream().native_handle());
    unsigned flags = 0;
    CHECK(cudaStreamGetFlags(runtime.transfer_stream().native_handle(), &flags) == cudaSuccess);
    CHECK(flags & cudaStreamNonBlocking);
    PinnedBuffer input(n * bytes), output(n * bytes);
    cudaPointerAttributes attributes{};
    CHECK(cudaPointerGetAttributes(&attributes, input.data()) == cudaSuccess);
    CHECK(attributes.type == cudaMemoryTypeHost);
    std::vector<BlockHandle> blocks;
    for (std::size_t i = 0; i < n; ++i) blocks.push_back(pool.allocate());
    // 故意打乱物理地址，验证 host 的第 i 块映射到 blocks[i]，而不是 id=i。
    std::mt19937 random(13);
    std::shuffle(blocks.begin(), blocks.end(), random);
    std::vector<unsigned char> expected(n * bytes), actual(n * bytes);
    for (int round = 0; round < 3; ++round) {
        for (auto& byte : expected) byte = static_cast<unsigned char>(random());
        input.copy_from(expected.data(), expected.size());
        runtime.write_batch(blocks, input);
        CHECK(input.busy() && runtime.pending_batches() == 1);
        CHECK(pool.ref_count(blocks[0]) == 2);
        throws<std::logic_error>([&] { input.data(); });
        throws<std::logic_error>([&] { input.copy_from(expected.data(), expected.size()); });
        throws<std::logic_error>([&] { pool.publish(blocks[0], Tokens(16, 1)); });
        throws<std::logic_error>([&] { pool.read_block(blocks[0], actual.data(), bytes); });
        throws<std::logic_error>([&] { runtime.read_batch(blocks, output); });
        runtime.synchronize();
        CHECK(!input.busy() && pool.ref_count(blocks[0]) == 1);
        runtime.read_batch(blocks, output);
        CHECK(output.busy());
        throws<std::logic_error>([&] { output.copy_to(actual.data(), actual.size()); });
        runtime.synchronize();
        output.copy_to(actual.data(), actual.size());
        CHECK(expected == actual);
    }
    throws<std::invalid_argument>([&] { runtime.write_batch({}, input); });
    PinnedBuffer wrong(1);
    throws<std::invalid_argument>([&] { runtime.write_batch(blocks, wrong); });
    if (n > 1) {
        auto invalid = blocks;
        invalid.back() = invalid.front();
        throws<std::invalid_argument>([&] { runtime.write_batch(invalid, input); });
        CHECK(runtime.pending_batches() == 0 && pool.ref_count(blocks[0]) == 1);
        invalid.back() = {9999, 1};
        throws<std::invalid_argument>([&] { runtime.write_batch(invalid, input); });
        CHECK(!input.busy() && runtime.pending_batches() == 0);
    }
    for (auto h : blocks) pool.release(h);
}

void lifetime_and_events() {
    GpuMemoryPool pool(2, 1024, 1);
    auto a = pool.allocate(), b = pool.allocate();
    std::vector<unsigned char> expected(1024, 83), output(1024);
    {
        AsyncTransferRuntime runtime(pool);
        auto pinned = std::make_unique<PinnedBuffer>(1024);
        pinned->copy_from(expected.data(), expected.size());
        runtime.write_block(a, *pinned);
        pinned.reset(); // runtime 仍持有底层 storage，DMA 不会访问已释放的 host 内存。
        PinnedBuffer second(1024);
        second.copy_from(expected.data(), expected.size());
        runtime.write_block(b, second);
        CHECK(runtime.pending_batches() == 2);
        pool.release(b); // 内部引用阻止在途块被重新分配。
        throws<CapacityError>([&] { pool.allocate(); });
        PinnedBuffer event_output(1024);
        runtime.compute_wait_for_transfers();
        CHECK(cudaMemcpyAsync(event_output.data(), pool.device_address(a), 1024,
                              cudaMemcpyDeviceToHost, runtime.compute_stream().native_handle()) == cudaSuccess);
        runtime.compute_stream().synchronize();
        event_output.copy_to(output.data(), output.size());
        CHECK(output == expected);
        runtime.synchronize();
        CHECK(pool.remaining_blocks() == 1);
        auto reused = pool.allocate();
        CHECK(reused.id == b.id && reused.generation != b.generation);
        throws<std::invalid_argument>([&] { runtime.write_block(b, second); });
        runtime.write_block(reused, second);
        pool.release(reused);
        // second 比 runtime 先析构，也由内部 shared storage 保护。
    }
    CHECK(pool.remaining_blocks() == 1);
    pool.read_block(a, output.data(), output.size());
    CHECK(output == expected);
    pool.publish(a, {1});
    PinnedBuffer pinned(1024);
    AsyncTransferRuntime runtime(pool);
    throws<std::invalid_argument>([&] { runtime.write_block(a, pinned); });
    pool.release(a);
}

int main() {
    try {
        int devices = 0;
        auto result = cudaGetDeviceCount(&devices);
        if (result == cudaErrorNoDevice || (result == cudaSuccess && devices == 0)) return 77;
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        throws<std::invalid_argument>([] { PinnedBuffer buffer(0); });
        for (std::size_t n : {1, 2, 4, 8, 16, 32}) batch_roundtrip(n);
        lifetime_and_events();
        std::cout << "Pinned/async: batch 1,2,4,8,16,32 and lifetime/event checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

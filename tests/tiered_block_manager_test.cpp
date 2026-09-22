#include "kvflux/tiered_block_manager.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <chrono>
#include <limits>
#include <utility>
#include <sstream>

#define CHECK(c) do { if (!(c)) throw std::runtime_error("check failed: " #c); } while (false)
template<class Error, class F> void throws(F fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}
using namespace kvflux;

void pressure_and_data() {
    TieredBlockManager manager(8, 12, 4096, 16);
    std::vector<LogicalBlockHandle> blocks;
    std::vector<std::vector<unsigned char>> data(12, std::vector<unsigned char>(4096));
    std::mt19937 random(161718);
    for (auto& block : data) for (auto& byte : block) byte = static_cast<unsigned char>(random());
    for (std::size_t i = 0; i < 12; ++i) blocks.push_back(manager.create(data[i].data(), data[i].size()));
    auto s = manager.stats();
    CHECK(s.gpu_resident == 8 && s.cpu_resident == 4 && s.logical_blocks == 12);
    CHECK(s.offloads == 4 && s.gpu_free == 0 && s.cpu_reserved_bytes == 12 * 4096);
    // 持有请求引用仍可迁移；最早使用的 0..3 被搬到 CPU，而非删除内容。
    for (std::size_t i = 0; i < 12; ++i) {
        CHECK(manager.ref_count(blocks[i]) == 1);
        CHECK(manager.location(blocks[i]) == (i < 4 ? BlockLocation::CPU_RESIDENT : BlockLocation::GPU_RESIDENT));
    }
    throws<CapacityError>([&] { manager.create(data[0].data(), data[0].size()); });
    manager.offload(blocks[4]);
    CHECK(manager.location(blocks[4]) == BlockLocation::TRANSFERRING);
    CHECK(manager.stats().gpu_free == 0); // D2H 完成之前绝不能复用槽位。
    manager.wait();
    CHECK(manager.stats().gpu_free == 1 && manager.location(blocks[4]) == BlockLocation::CPU_RESIDENT);
    manager.load(blocks[0]);
    CHECK(manager.location(blocks[0]) == BlockLocation::TRANSFERRING);
    manager.wait();
    CHECK(manager.location(blocks[0]) == BlockLocation::GPU_RESIDENT);
    std::vector<std::size_t> order(12);
    std::iota(order.begin(), order.end(), 0);
    std::vector<unsigned char> result(4096);
    for (int pass = 0; pass < 4; ++pass) {
        std::shuffle(order.begin(), order.end(), random);
        for (auto i : order) {
            manager.read_block(blocks[i], result.data(), result.size());
            CHECK(result == data[i]);
        }
    }
    for (auto h : blocks) manager.release(h);
    CHECK(manager.stats().logical_blocks == 0 && manager.stats().gpu_free == 8);
    auto replacement = manager.create(data[0].data(), data[0].size());
    CHECK(replacement.id == blocks.back().id && replacement.generation != blocks.back().generation);
    throws<std::invalid_argument>([&] { manager.location(blocks.back()); });
    manager.release(replacement);
}

void leases_and_lru() {
    TieredBlockManager manager(2, 4, 1024, 1);
    std::vector<unsigned char> data(1024, 7), changed(1024, 29), output(1024);
    auto a = manager.create(data.data(), data.size());
    auto b = manager.create(data.data(), data.size());
    {
        auto lease = manager.acquire_gpu(a); // a 被保护，LRU 只能选择 b。
        CHECK(cudaMemcpy(lease.data(), changed.data(), changed.size(), cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        auto c = manager.create(data.data(), data.size());
        CHECK(manager.location(b) == BlockLocation::CPU_RESIDENT);
        throws<std::logic_error>([&] { manager.offload(a); });
        {
            auto second = manager.acquire_gpu(c);
            throws<CapacityError>([&] { manager.load(b); });
            CHECK(manager.location(b) == BlockLocation::CPU_RESIDENT);
        }
        manager.release(c);
    }
    manager.offload(a); // 外部 kernel/写入改变了 GPU，offload 必须带回最新值。
    manager.wait();
    manager.read_block(a, output.data(), output.size());
    CHECK(output == changed);
    {
        auto lease = manager.acquire_gpu(a);
        manager.release(a); // lease 独立持有引用，数据不会被提前销毁。
        CHECK(manager.ref_count(a) == 1);
    }
    throws<std::invalid_argument>([&] { manager.location(a); });
    manager.release(b);
}

void prefetch_and_lifecycle() {
    TieredBlockManager manager(3, 6, 4096, 1);
    std::vector<LogicalBlockHandle> blocks;
    std::vector<unsigned char> data(4096, 63), output(4096);
    for (int i = 0; i < 6; ++i) blocks.push_back(manager.create(data.data(), data.size()));
    auto current = manager.acquire_gpu(blocks[0]); // GPU 仍满，next 2 需要真实 offload 后再 load。
    CHECK(manager.prefetch_next(blocks, 0, 2) == 2);
    CHECK(manager.location(blocks[1]) == BlockLocation::TRANSFERRING);
    CHECK(manager.location(blocks[2]) == BlockLocation::TRANSFERRING);
    CHECK(manager.location(blocks[0]) == BlockLocation::GPU_RESIDENT);
    CHECK(manager.prefetch_next(blocks, 0, 2) == 0); // 重复预取不会重复分配。
    manager.wait();
    CHECK(manager.location(blocks[1]) == BlockLocation::GPU_RESIDENT);
    CHECK(manager.location(blocks[2]) == BlockLocation::GPU_RESIDENT);
    manager.read_block(blocks[1], output.data(), output.size());
    CHECK(output == data);
    CHECK(manager.prefetch_next(blocks, 0, 5) == 0); // 全窗口受保护，不能互相驱逐。
    CHECK(manager.stats().prefetch_skipped == 3);
    throws<std::out_of_range>([&] { manager.prefetch_next(blocks, 6, 1); });
    auto invalid = blocks;
    invalid[1] = {999, 1};
    throws<std::invalid_argument>([&] { manager.prefetch_next(invalid, 0, 2); });
    manager.offload(blocks[1]);
    manager.release(blocks[1]); // 最后一个引用释放时必须先完成 DMA。
    throws<std::invalid_argument>([&] { manager.location(blocks[1]); });
    for (std::size_t i = 0; i < blocks.size(); ++i) if (i != 1) manager.release(blocks[i]);
    // current lease 最后析构，届时归还 blocks[0] 的最后一份引用。
}

void queued_prefetch_and_poll() {
    TieredBlockManager manager(2, 4, 4096, 1);
    std::vector<unsigned char> data(4096, 81), output(4096);
    std::vector<LogicalBlockHandle> blocks;
    for (int i = 0; i < 4; ++i) blocks.push_back(manager.create(data.data(), data.size()));
    {
        auto current = manager.acquire_gpu(blocks[0]);
        const auto before = manager.stats();
        CHECK(manager.prefetch_next(blocks, 0, 0) == 0);
        CHECK(manager.stats().offloads == before.offloads);
        CHECK(manager.prefetch_next(blocks, 0, 1) == 1);
        // 只用 poll 推进 D2H -> 释放槽位 -> 排队 H2D -> GPU resident。
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!manager.poll()) CHECK(std::chrono::steady_clock::now() < deadline);
        CHECK(manager.location(blocks[1]) == BlockLocation::GPU_RESIDENT);
        manager.read_block(blocks[1], output.data(), output.size());
        CHECK(output == data);
        CHECK(manager.stats().transferring == 0);
        CHECK(manager.stats().gpu_resident == 2);
        auto moved = std::move(current);
        CHECK(current.data() == nullptr && moved.data() != nullptr);
        throws<std::logic_error>([&] { manager.offload(blocks[0]); });
    }
    {
        auto current = manager.acquire_gpu(blocks[2]);
        CHECK(manager.prefetch_next(blocks, 2, std::numeric_limits<std::size_t>::max()) == 1);
        // 最后一个请求引用可在预取尚未拿到 GPU 槽位时释放。
        manager.release(blocks[3]);
        CHECK(manager.stats().transferring == 0);
        throws<std::invalid_argument>([&] { manager.load(blocks[3]); });
        CHECK(manager.stats().gpu_free == 1);
    }
    const std::vector<LogicalBlockHandle> repeated{blocks[0], blocks[1], blocks[1]};
    manager.offload(blocks[1]);
    manager.wait();
    CHECK(manager.prefetch_next(repeated, 0, 2) == 1);
    manager.load(blocks[1]); // 与预取中的 demand 合并。
    manager.wait();
    CHECK(manager.stats().gpu_resident <= 2);
    for (std::size_t i = 0; i < 3; ++i) manager.release(blocks[i]);
    CHECK(manager.stats().logical_blocks == 0 && manager.stats().gpu_free == 2);
}

void lru_touch_and_failed_create() {
    TieredBlockManager manager(2, 3, 256, 1);
    std::vector<unsigned char> data(256, 19);
    auto a = manager.create(data.data(), data.size());
    auto b = manager.create(data.data(), data.size());
    manager.load(a); // 无 lease，单纯访问也应更新 LRU。
    auto c = manager.create(data.data(), data.size());
    CHECK(manager.location(b) == BlockLocation::CPU_RESIDENT);
    manager.release(b);
    {
        auto first = manager.acquire_gpu(a);
        auto nested = manager.acquire_gpu(a);
        auto second = manager.acquire_gpu(c);
        throws<CapacityError>([&] { manager.create(data.data(), data.size()); });
        CHECK(manager.stats().logical_blocks == 2); // 失败创建不能泄漏逻辑槽位。
        CHECK(manager.stats().gpu_resident == 2 && manager.stats().transferring == 0);
        CHECK(manager.ref_count(a) == 3);
        CHECK(manager.prefetch_next({a, c}, 0, 1) == 0);
    }
    auto replacement = manager.create(data.data(), data.size());
    throws<std::invalid_argument>([&] { manager.retain(b); });
    manager.offload(replacement);
    manager.offload(replacement); // 重复 offload 合并；反方向 demand 等待后再 load。
    manager.load(replacement);
    manager.wait();
    CHECK(manager.location(replacement) == BlockLocation::GPU_RESIDENT);
    manager.release(a);
    manager.release(c);
    manager.release(replacement);
    CHECK(manager.stats().gpu_free == 2);
}

void metrics_accounting() {
    constexpr std::size_t bytes = 4096;
    TieredBlockManager manager(2, 4, bytes, 1);
    CHECK(manager.metrics().gpu_blocks_free == 2);
    CHECK(manager.metrics().transfers.cpu_to_gpu.bandwidth_gbps() == 0);
    std::vector<unsigned char> data(bytes, 12);
    auto a = manager.create(data.data(), bytes);
    auto b = manager.create(data.data(), bytes);
    auto c = manager.create(data.data(), bytes);
    auto initial = manager.metrics();
    CHECK(initial.gpu_blocks_total == 2 && initial.gpu_blocks_used == 2 && initial.gpu_blocks_free == 0);
    CHECK(initial.cpu_cached_blocks == 1 && initial.offload_count == 1 && initial.load_count == 3);
    CHECK(initial.transfers.gpu_to_cpu.bytes == bytes && initial.transfers.cpu_to_gpu.bytes == 3 * bytes);
    CHECK(initial.demand_count == 0); // 初始化不属于 demand。
    manager.offload(b);
    auto during = manager.metrics();
    CHECK(during.gpu_blocks_used == 2 && during.transferring_blocks == 1);
    CHECK(during.offload_count == 1 && during.transfers.gpu_to_cpu.bytes == bytes);
    manager.wait();
    CHECK(manager.prefetch_next({c, a}, 0, 1) == 1); // free-slot H2D。
    CHECK(manager.metrics().prefetch_count == 1 && manager.metrics().prefetch_outstanding == 1);
    manager.wait();
    { auto lease = manager.acquire_gpu(a); }
    auto ready = manager.metrics();
    CHECK(ready.prefetch_hits == 1 && ready.prefetch_misses == 0 && ready.prefetch_outstanding == 0);
    { auto lease = manager.acquire_gpu(a); } // 同一预取只算一次 hit。
    CHECK(manager.metrics().prefetch_hits == 1 && manager.metrics().gpu_demand_hits == 2);
    CHECK(manager.prefetch_next({a, b}, 0, 1) == 1); // 满池，先排队等待 c 搬出。
    { auto lease = manager.acquire_gpu(b); } // 尚未 poll：late，属于 miss。
    auto late = manager.metrics();
    CHECK(late.prefetch_count == 2 && late.prefetch_late == 1 && late.prefetch_misses == 1);
    CHECK(late.prefetch_hits == 1 && late.prefetch_unused == 0);
    CHECK(late.demand_count == 3 && late.request_stall_ms > 0);
    CHECK(manager.prefetch_next({b, c}, 0, 1) == 1);
    manager.wait();
    manager.offload(c); // 预取完成但未使用即被搬出，不能算 hit。
    manager.wait();
    auto unused = manager.metrics();
    CHECK(unused.prefetch_unused == 1);
    CHECK(unused.prefetch_count == unused.prefetch_hits + unused.prefetch_late + unused.prefetch_unused);
    manager.release(a);
    manager.release(b);
    manager.release(c);
    auto final = manager.metrics();
    CHECK(final.gpu_blocks_used == 0 && final.gpu_blocks_free == 2 && final.cpu_cached_blocks == 0);
    CHECK(final.transfers.gpu_to_cpu.bytes == final.offload_count * bytes);
    CHECK(final.transfers.cpu_to_gpu.bytes == final.load_count * bytes);
    std::ostringstream text;
    text << final;
    CHECK(text.str().find("gpu_blocks_total=2\n") != std::string::npos);
    CHECK(text.str().find("prefetch_hits=1\n") != std::string::npos);
}

int main() {
    try {
        int devices = 0;
        const auto result = cudaGetDeviceCount(&devices);
        if (result == cudaErrorNoDevice || (result == cudaSuccess && !devices)) return 77;
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        throws<std::invalid_argument>([] { TieredBlockManager m(2, 1, 16, 1); });
        throws<std::overflow_error>([] { TieredBlockManager m(1, 2, std::numeric_limits<std::size_t>::max(), 1); });
        pressure_and_data();
        leases_and_lru();
        prefetch_and_lifecycle();
        queued_prefetch_and_poll();
        lru_touch_and_failed_create();
        metrics_accounting();
        std::cout << "Tiered cache: pressure, roundtrip, LRU, leases and prefetch passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

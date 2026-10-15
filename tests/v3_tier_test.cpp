#include "kvflux/v3/tier.h"

#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Fn> void capacity_error(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const kvflux::CapacityError&) { caught = true; }
    CHECK(caught);
}

template<class Fn> void invalid_argument(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
}

void slot_tier_accounting_and_reuse() {
    kvflux::v3::CpuTier tier(2, 1 << 20);
    CHECK(tier.kind() == kvflux::v3::StorageTier::CPU);
    CHECK(tier.capacity() == 2 && tier.available() == 2 && tier.allocated_chunks() == 0);
    CHECK(tier.used_bytes() == 0 && tier.available_bytes() == 2 * (1 << 20));

    const auto a = tier.allocate();
    const auto b = tier.allocate();
    CHECK(tier.available() == 0 && tier.allocated_chunks() == 2);
    CHECK(tier.used_bytes() == 2 * (1 << 20) && tier.available_bytes() == 0);
    capacity_error([&] { tier.allocate(); });

    // 释放后同一槽位可被重用，但旧 generation 的位置已失效。
    tier.release(a);
    CHECK(tier.available() == 1 && tier.allocated_chunks() == 1);
    const auto c = tier.allocate();
    CHECK(c.slot == a.slot && c.generation != a.generation);
    invalid_argument([&] { tier.release(a); }); // 旧 generation 被拒绝
    tier.release(b);
    tier.release(c);
    CHECK(tier.allocated_chunks() == 0 && tier.available() == 2);
}

void slot_tier_rejects_double_release() {
    kvflux::v3::NvmeTier tier(1, 4096);
    CHECK(tier.kind() == kvflux::v3::StorageTier::NVME);
    const auto a = tier.allocate();
    tier.release(a);
    invalid_argument([&] { tier.release(a); }); // 重复释放
    invalid_argument([&] { tier.release({a.slot, a.generation + 7}); }); // 失配 generation
    invalid_argument([&] { tier.release({999, 1}); }); // 越界槽位
    CHECK(tier.available() == 1);
}

void gpu_tier_wraps_physical_block_pool() {
    kvflux::PhysicalBlockPool pool(3, 16);
    kvflux::v3::GpuTier tier(pool, 1 << 20);
    CHECK(tier.kind() == kvflux::v3::StorageTier::GPU);
    CHECK(tier.capacity() == 3 && tier.available() == 3);

    const auto a = tier.allocate();
    const auto b = tier.allocate();
    CHECK(tier.allocated_chunks() == 2 && tier.available() == 1);
    // slot 就是 PhysicalBlockID。
    CHECK(a.slot < 3 && b.slot < 3 && a.slot != b.slot);

    tier.release(a);
    CHECK(tier.allocated_chunks() == 1 && tier.available() == 2);
    // 旧位置被池的 generation 校验拒绝。
    invalid_argument([&] { tier.release(a); });

    // 复用刚释放的物理编号，generation 递增。
    const auto c = tier.allocate();
    CHECK(c.slot == a.slot && c.generation != a.generation);

    const auto d = tier.allocate();
    capacity_error([&] { tier.allocate(); });
    tier.release(b);
    tier.release(c);
    tier.release(d);
    CHECK(tier.allocated_chunks() == 0 && tier.available() == 3);
    CHECK(pool.available() == pool.capacity());
}

int main() {
    try {
        slot_tier_accounting_and_reuse();
        slot_tier_rejects_double_release();
        gpu_tier_wraps_physical_block_pool();
        std::cout << "V3 tier: accounting, generation checks, pool reuse\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

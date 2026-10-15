#pragma once

#include "kvflux/physical_block_pool.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kvflux::v3 {

enum class StorageTier { GPU, CPU, NVME };

// 某层存储中一个 chunk 槽位的位置。slot 只是下标；generation 区分同一槽位
// 前后两次分配，目录之外保存的旧位置会因 generation 失配而被拒绝。
struct TierLocation {
    std::size_t slot;
    std::uint64_t generation;
    bool operator==(const TierLocation& other) const noexcept {
        return slot == other.slot && generation == other.generation;
    }
};

// 单层存储的元数据接口。V3.0 只管理槽位占用与容量统计，不传输任何数据。
// 每个槽位固定容纳一个 KVChunk；与 v0/v2 一样由调用方保证单线程访问。
class Tier {
public:
    virtual ~Tier() = default;

    virtual StorageTier kind() const noexcept = 0;
    virtual std::size_t chunk_bytes() const noexcept = 0;
    // 槽位总数与当前持有分配的槽位数。
    virtual std::size_t capacity() const noexcept = 0;
    virtual std::size_t allocated_chunks() const noexcept = 0;
    // 可立即分配的槽位数。
    virtual std::size_t available() const noexcept = 0;
    virtual std::size_t used_bytes() const noexcept = 0;
    virtual std::size_t available_bytes() const noexcept = 0;

    // 返回一个活跃位置。没有可分配槽位时抛 CapacityError。
    virtual TierLocation allocate() = 0;
    // 位置必须仍活跃；重复释放或归还旧 generation 会抛 invalid_argument。
    virtual void release(const TierLocation& location) = 0;
};

// CPU/NVMe 共用的定长槽位池：free list + generation 校验，语义与 v0 一致。
// 真实 pinned 显存与文件 I/O 分别在后续阶段接入，本版只做元数据。
class SlotTier : public Tier {
public:
    SlotTier(StorageTier kind, std::size_t chunk_slots, std::size_t chunk_bytes);

    StorageTier kind() const noexcept override { return kind_; }
    std::size_t chunk_bytes() const noexcept override { return chunk_bytes_; }
    std::size_t capacity() const noexcept override { return slots_.size(); }
    std::size_t allocated_chunks() const noexcept override { return allocated_; }
    std::size_t available() const noexcept override { return slots_.size() - allocated_; }
    std::size_t used_bytes() const noexcept override { return allocated_ * chunk_bytes_; }
    std::size_t available_bytes() const noexcept override {
        return available() * chunk_bytes_;
    }

    TierLocation allocate() override;
    void release(const TierLocation& location) override;

private:
    static constexpr std::size_t none = static_cast<std::size_t>(-1);
    struct Slot {
        std::uint64_t generation = 0;
        bool free = true;
        std::size_t next_free = none;
    };

    StorageTier kind_;
    std::size_t chunk_bytes_;
    std::vector<Slot> slots_;
    std::size_t free_head_ = 0;
    std::size_t allocated_ = 0;
};

// V1 的 pinned memory 管理组件依赖 CUDA；V3.0 不传输数据，先用槽位元数据
// 占位，接入真实 pinned 分配时保持接口不变。
class CpuTier : public SlotTier {
public:
    CpuTier(std::size_t chunk_slots, std::size_t chunk_bytes)
        : SlotTier(StorageTier::CPU, chunk_slots, chunk_bytes) {}
};

// 槽位代表一个固定大小的文件块编号；真实文件分配与读写属 V3.2。
class NvmeTier : public SlotTier {
public:
    NvmeTier(std::size_t chunk_slots, std::size_t chunk_bytes)
        : SlotTier(StorageTier::NVME, chunk_slots, chunk_bytes) {}
};

// 复用 V2 的 PhysicalBlockPool：分配/归还经池的 free list 与 generation 校验，
// slot 即 PhysicalBlockID。本类不发布前缀，只借池做槽位元数据管理。
class GpuTier : public Tier {
public:
    GpuTier(PhysicalBlockPool& pool, std::size_t chunk_bytes);

    StorageTier kind() const noexcept override { return StorageTier::GPU; }
    std::size_t chunk_bytes() const noexcept override { return chunk_bytes_; }
    std::size_t capacity() const noexcept override;
    std::size_t allocated_chunks() const noexcept override { return allocated_; }
    std::size_t available() const noexcept override;
    std::size_t used_bytes() const noexcept override { return allocated_ * chunk_bytes_; }
    std::size_t available_bytes() const noexcept override {
        return available() * chunk_bytes_;
    }

    TierLocation allocate() override;
    void release(const TierLocation& location) override;

private:
    PhysicalBlockPool& pool_; // pool 必须比 GpuTier 活得更久。
    std::size_t chunk_bytes_;
    std::size_t allocated_ = 0;
};

} // namespace kvflux::v3

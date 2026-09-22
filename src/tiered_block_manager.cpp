#include "kvflux/tiered_block_manager.h"
#include "cuda_utils.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace kvflux {
namespace {
std::size_t checked_capacity(std::size_t cpu, std::size_t gpu, std::size_t bytes) {
    if (!gpu || cpu < gpu || !bytes) throw std::invalid_argument("require 0 < GPU blocks <= CPU backing blocks");
    if (cpu > std::numeric_limits<std::size_t>::max() / bytes) throw std::overflow_error("CPU backing size overflow");
    return cpu;
}
}
TieredBlockManager::TieredBlockManager(std::size_t gpu, std::size_t cpu, std::size_t bytes,
                                       std::size_t tokens, int device)
    : logical_(checked_capacity(cpu, gpu, bytes), tokens), gpu_(gpu, bytes, tokens, device),
      transfers_(gpu_), entries_(cpu), resident_lru_(cpu), cpu_reserved_bytes_(cpu * bytes) {
    backing_.reserve(cpu);
    pending_.reserve(cpu);
    queued_.reserve(cpu);
    dispatch_work_.reserve(cpu);
    for (std::size_t i = 0; i < cpu; ++i) backing_.push_back(std::make_unique<PinnedBuffer>(bytes));
}
TieredBlockManager::~TieredBlockManager() noexcept { try { wait(); } catch (...) {} }

const TieredBlockManager::Entry& TieredBlockManager::checked(LogicalBlockHandle h) const {
    (void)logical_.ref_count(raw(h)); // 同时检查范围、generation 和释放后的旧句柄。
    return entries_[h.id];
}
TieredBlockManager::Entry& TieredBlockManager::checked(LogicalBlockHandle h) {
    return const_cast<Entry&>(std::as_const(*this).checked(h));
}
BlockLocation TieredBlockManager::location(LogicalBlockHandle h) const { return checked(h).location; }
std::size_t TieredBlockManager::ref_count(LogicalBlockHandle h) const { return logical_.ref_count(raw(h)); }
void TieredBlockManager::retain(LogicalBlockHandle h) { (void)checked(h); logical_.retain(raw(h)); }

LogicalBlockHandle TieredBlockManager::create(const void* host, std::size_t bytes) {
    if (!host || bytes != block_bytes()) throw std::invalid_argument("create requires one full host block");
    auto slot = logical_.allocate();
    LogicalBlockHandle h{slot.id, slot.generation};
    auto& entry = entries_[h.id];
    entry = Entry{};
    entry.live = true;
    entry.handle = h;
    try {
        backing_[h.id]->copy_from(host, bytes);
        load(h);
        wait();
    } catch (...) {
        release(h);
        throw;
    }
    return h;
}

void TieredBlockManager::release(LogicalBlockHandle h) {
    auto& e = checked(h);
    if (ref_count(h) == 1) {
        if (e.location == BlockLocation::TRANSFERRING) wait();
        discard_prefetch(e);
        resident_lru_.erase(h.id);
        if (e.gpu) gpu_.release(*e.gpu);
        e = Entry{};
    }
    logical_.release(raw(h));
}

TieredBlockManager::GpuLease::~GpuLease() noexcept { if (owner_) owner_->unpin(handle_); }
TieredBlockManager::GpuLease::GpuLease(GpuLease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), handle_(other.handle_), address_(other.address_) {
    other.address_ = nullptr;
}
void TieredBlockManager::unpin(LogicalBlockHandle h) noexcept {
    auto& e = entries_[h.id];
    if (--e.pins == 0) resident_lru_.touch(h.id);
    release(h);
}
TieredBlockManager::GpuLease TieredBlockManager::acquire_gpu(LogicalBlockHandle h) {
    const auto begin = std::chrono::steady_clock::now();
    const auto& before = checked(h);
    const bool resident = before.location == BlockLocation::GPU_RESIDENT;
    const bool prefetched = before.prefetched;
    load(h);
    if (checked(h).location == BlockLocation::TRANSFERRING) wait();
    auto& e = checked(h);
    if (e.pins == std::numeric_limits<std::size_t>::max()) throw std::overflow_error("GPU lease count overflow");
    retain(h); // lease 自己持有逻辑引用，调用方可提前释放请求引用。
    ++e.pins;
    resident_lru_.erase(h.id);
    ++demand_count_;
    if (resident) ++gpu_demand_hits_; else ++prefetch_misses_;
    if (prefetched) {
        if (resident) ++prefetch_hits_; else ++prefetch_late_;
        e.prefetched = false; // 每次接纳的预取至多被一个 demand 消费。
    }
    request_stall_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    return GpuLease(this, h, gpu_.device_address(*e.gpu));
}
void TieredBlockManager::discard_prefetch(Entry& e) noexcept {
    if (e.prefetched) { ++prefetch_unused_; e.prefetched = false; }
}
void TieredBlockManager::read_block(LogicalBlockHandle h, void* host, std::size_t bytes) {
    if (!host || bytes != block_bytes()) throw std::invalid_argument("read requires one full host block");
    auto lease = acquire_gpu(h);
    gpu_.read_block(*checked(h).gpu, host, bytes);
}

std::size_t TieredBlockManager::victim(const std::vector<bool>& protected_ids) const noexcept {
    auto id = resident_lru_.oldest();
    while (id != detail::IndexLru::none && !protected_ids.empty() && protected_ids[id]) id = resident_lru_.next(id);
    return id;
}
void TieredBlockManager::make_room() {
    if (gpu_.remaining_blocks()) return;
    wait(); // 已提交的迁移可能释放空间，也要兑现之前的预取槽位预约。
    if (gpu_.remaining_blocks()) return;
    const auto id = victim({});
    if (id == detail::IndexLru::none) throw CapacityError(); // 全部被 kernel lease 保护。
    start_offload(entries_[id].handle);
    wait(); // GPU 槽位只能在 D2H 完成之后释放。
}

void TieredBlockManager::start_load(LogicalBlockHandle h) {
    auto& e = checked(h);
    e.gpu = gpu_.allocate();
    e.location = BlockLocation::TRANSFERRING;
    e.move = Move::ToGpu;
    pending_.push_back(h);
    try { transfers_.write_block(*e.gpu, *backing_[h.id]); }
    catch (...) {
        try { transfers_.synchronize(); } catch (...) {}
        finish(false);
        throw;
    }
}
void TieredBlockManager::start_offload(LogicalBlockHandle h) {
    auto& e = checked(h);
    resident_lru_.erase(h.id);
    e.location = BlockLocation::TRANSFERRING;
    e.move = Move::ToCpu;
    pending_.push_back(h);
    try {
        transfers_.read_block(*e.gpu, *backing_[h.id]);
        discard_prefetch(e);
    }
    catch (...) {
        try { transfers_.synchronize(); } catch (...) {}
        finish(false);
        throw;
    }
}
void TieredBlockManager::offload(LogicalBlockHandle h) {
    auto& e = checked(h);
    if (e.pins) throw std::logic_error("cannot offload a GPU-leased block");
    if (e.move == Move::ToCpu) return;
    if (e.location == BlockLocation::TRANSFERRING) wait();
    if (e.location == BlockLocation::GPU_RESIDENT) start_offload(h);
}
void TieredBlockManager::load(LogicalBlockHandle h) {
    auto& e = checked(h);
    if (e.move == Move::ToCpu) wait();
    if (e.location == BlockLocation::GPU_RESIDENT) {
        if (!e.pins) resident_lru_.touch(h.id);
        return;
    }
    if (e.location == BlockLocation::TRANSFERRING) return;
    make_room();
    start_load(h);
}

void TieredBlockManager::finish(bool success) noexcept {
    for (auto h : pending_) {
        auto& e = entries_[h.id];
        if (!success) discard_prefetch(e);
        const bool to_cpu = e.move == Move::ToCpu;
        if ((success && to_cpu) || (!success && !to_cpu)) {
            gpu_.release(*e.gpu);
            e.gpu.reset();
            e.location = BlockLocation::CPU_RESIDENT;
        } else {
            e.location = BlockLocation::GPU_RESIDENT;
            resident_lru_.touch(h.id);
        }
        if (success) { if (to_cpu) ++offloads_; else ++loads_; }
        e.move = Move::None;
    }
    pending_.clear();
    if (!success) {
        for (auto h : queued_) {
            auto& e = entries_[h.id];
            discard_prefetch(e);
            e.location = BlockLocation::CPU_RESIDENT;
            e.move = Move::None;
        }
        queued_.clear();
    }
}
void TieredBlockManager::dispatch_queued() {
    // 每个 queued load 都对应一个已成功 offload 的槽位，不能超额预约。
    // 先移到局部表，避免失败时 finish(false) 把已经装入的槽位误当作未提交。
    dispatch_work_.swap(queued_); // 两个 vector 均已预留容量，迁移热路径不再分配。
    std::size_t i = 0;
    try {
        for (; i < dispatch_work_.size(); ++i) {
            auto& e = entries_[dispatch_work_[i].id];
            e.location = BlockLocation::CPU_RESIDENT;
            e.move = Move::None;
            start_load(dispatch_work_[i]);
        }
    } catch (...) {
        for (; i < dispatch_work_.size(); ++i) {
            auto& e = entries_[dispatch_work_[i].id];
            discard_prefetch(e);
            e.location = BlockLocation::CPU_RESIDENT;
            e.move = Move::None;
        }
        dispatch_work_.clear();
        throw;
    }
    dispatch_work_.clear();
}
void TieredBlockManager::wait() {
    while (!pending_.empty() || !queued_.empty()) {
        try { transfers_.synchronize(); }
        catch (...) { finish(false); throw; }
        finish(true);
        dispatch_queued();
    }
}
bool TieredBlockManager::poll() {
    if (pending_.empty() && queued_.empty()) return true;
    detail::DeviceGuard guard(device());
    const auto result = cudaStreamQuery(transfers_.transfer_stream().native_handle());
    if (result == cudaErrorNotReady) return false;
    // synchronize 统一执行 CUDA 错误处理和 runtime 引用回收；成功查询时不会等待。
    try { transfers_.synchronize(); }
    catch (...) { finish(false); throw; }
    finish(true);
    dispatch_queued();
    return pending_.empty();
}

std::size_t TieredBlockManager::prefetch_next(const std::vector<LogicalBlockHandle>& order,
                                            std::size_t current, std::size_t next_n) {
    if (current >= order.size()) throw std::out_of_range("prefetch current index out of range");
    const auto count = std::min(next_n, order.size() - current - 1);
    std::vector<bool> protected_ids(entries_.size(), false);
    for (std::size_t offset = 0; offset <= count; ++offset) {
        const auto h = order[current + offset];
        (void)checked(h);
        protected_ids[h.id] = true;
    }
    poll();
    std::size_t submitted = 0;
    for (std::size_t offset = 1; offset <= count; ++offset) {
        const auto h = order[current + offset];
        auto& e = checked(h);
        if (e.location != BlockLocation::CPU_RESIDENT) continue;
        if (gpu_.remaining_blocks()) {
            start_load(h);
        } else {
            const auto id = victim(protected_ids);
            if (id == detail::IndexLru::none) { ++prefetch_skipped_; continue; }
            start_offload(entries_[id].handle);
            e.location = BlockLocation::TRANSFERRING;
            e.move = Move::QueuedLoad;
            queued_.push_back(h);
        }
        ++submitted;
        ++prefetch_submitted_;
        e.prefetched = true;
    }
    return submitted;
}
TieredBlockManager::Stats TieredBlockManager::stats() const noexcept {
    Stats s{0, 0, 0, 0, gpu_.remaining_blocks(), gpu_.total_blocks(), entries_.size(),
            cpu_reserved_bytes_, offloads_, loads_, prefetch_submitted_, prefetch_skipped_};
    for (const auto& e : entries_) if (e.live) {
        ++s.logical_blocks;
        if (e.location == BlockLocation::GPU_RESIDENT) ++s.gpu_resident;
        else if (e.location == BlockLocation::CPU_RESIDENT) ++s.cpu_resident;
        else ++s.transferring;
    }
    return s;
}
CacheMetrics TieredBlockManager::metrics() const noexcept {
    const auto s = stats();
    CacheMetrics m;
    m.gpu_blocks_total = s.gpu_capacity;
    m.gpu_blocks_free = s.gpu_free;
    m.gpu_blocks_used = s.gpu_capacity - s.gpu_free; // 含 DMA 正在占用的物理槽位。
    m.cpu_cached_blocks = s.cpu_resident;
    m.logical_blocks = s.logical_blocks;
    m.transferring_blocks = s.transferring;
    m.transfers = transfers_.metrics();
    m.offload_count = s.offloads;
    m.load_count = s.loads;
    m.prefetch_count = s.prefetch_submitted;
    m.prefetch_skipped = s.prefetch_skipped;
    m.prefetch_hits = prefetch_hits_;
    m.prefetch_misses = prefetch_misses_;
    m.prefetch_late = prefetch_late_;
    m.prefetch_unused = prefetch_unused_;
    for (const auto& e : entries_) if (e.live && e.prefetched) ++m.prefetch_outstanding;
    m.demand_count = demand_count_;
    m.gpu_demand_hits = gpu_demand_hits_;
    m.request_stall_ms = request_stall_ms_;
    return m;
}
} // namespace kvflux

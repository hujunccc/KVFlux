#include "kvflux/v3/gpu_cpu_transfer.h"

#include "../cuda_utils.h"

#include <stdexcept>

namespace kvflux::v3 {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::logic_error(message);
}

} // namespace

GpuCpuTransfer::GpuCpuTransfer(CacheDirectory& directory, GpuChunkStore& gpu_store,
                               HostChunkBuffers& cpu_buffers, int device)
    : directory_(directory), gpu_store_(gpu_store), cpu_buffers_(cpu_buffers),
      stream_(device) {
    if (gpu_store_.chunk_bytes() != cpu_buffers_.chunk_bytes()) {
        throw std::invalid_argument("GPU and CPU chunk sizes must match");
    }
}

GpuCpuTransfer::~GpuCpuTransfer() noexcept {
    // 目录不允许绕过 finish 移除在途副本；析构前收敛所有 event。
    for (auto& entry : pending_) {
        if (entry.second.done) cudaEventDestroy(entry.second.done);
    }
}

void GpuCpuTransfer::begin_offload(const CacheKey& key) {
    require(pending_.find(key) == pending_.end(), "transfer already pending");
    const auto& record = directory_.at(key);
    require(record.gpu.has_value() && record.gpu->state == ChunkState::READY,
            "offload requires a READY gpu replica");
    require(!record.cpu.has_value(), "cpu replica already exists");

    const auto source = record.gpu->location;
    directory_.set_state(key, StorageTier::GPU, ChunkState::COPYING);
    TierLocation destination{};
    cudaEvent_t done = nullptr;
    try {
        destination = directory_.place(key, StorageTier::CPU, ChunkState::COPYING);
        detail::check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming),
                      "cudaEventCreateWithFlags");
        detail::check(cudaMemcpyAsync(cpu_buffers_.data(destination.slot),
                                      gpu_store_.device_ptr(source),
                                      gpu_store_.chunk_bytes(), cudaMemcpyDeviceToHost,
                                      stream_.native_handle()),
                      "cudaMemcpyAsync D2H");
        detail::check(cudaEventRecord(done, stream_.native_handle()), "cudaEventRecord");
    } catch (...) {
        if (done) cudaEventDestroy(done);
        stream_.synchronize();
        if (directory_.at(key).cpu.has_value()) {
            directory_.set_state(key, StorageTier::CPU, ChunkState::ERROR);
            directory_.remove(key, StorageTier::CPU);
        }
        directory_.set_state(key, StorageTier::GPU, ChunkState::READY);
        throw;
    }
    directory_.pin(key);
    pending_.emplace(key, Pending{source, destination, true, done});
}

void GpuCpuTransfer::begin_onload(const CacheKey& key) {
    require(pending_.find(key) == pending_.end(), "transfer already pending");
    const auto& record = directory_.at(key);
    require(record.cpu.has_value() && record.cpu->state == ChunkState::READY,
            "onload requires a READY cpu replica");
    require(!record.gpu.has_value(), "gpu replica already exists");

    const auto source = record.cpu->location;
    directory_.set_state(key, StorageTier::CPU, ChunkState::COPYING);
    TierLocation destination{};
    cudaEvent_t done = nullptr;
    try {
        destination = directory_.place(key, StorageTier::GPU, ChunkState::LOADING);
        detail::check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming),
                      "cudaEventCreateWithFlags");
        detail::check(cudaMemcpyAsync(gpu_store_.device_ptr(destination),
                                      cpu_buffers_.data(source.slot),
                                      gpu_store_.chunk_bytes(), cudaMemcpyHostToDevice,
                                      stream_.native_handle()),
                      "cudaMemcpyAsync H2D");
        detail::check(cudaEventRecord(done, stream_.native_handle()), "cudaEventRecord");
    } catch (...) {
        if (done) cudaEventDestroy(done);
        stream_.synchronize();
        if (directory_.at(key).gpu.has_value()) {
            directory_.set_state(key, StorageTier::GPU, ChunkState::ERROR);
            directory_.remove(key, StorageTier::GPU);
        }
        directory_.set_state(key, StorageTier::CPU, ChunkState::READY);
        throw;
    }
    directory_.pin(key);
    pending_.emplace(key, Pending{source, destination, false, done});
}

void GpuCpuTransfer::finish(const CacheKey& key, bool offload, bool release_source) {
    const auto it = pending_.find(key);
    require(it != pending_.end(), "no pending transfer for this chunk");
    require(it->second.offload == offload, "pending transfer direction mismatch");
    const Pending pending = it->second;

    detail::check(cudaEventSynchronize(pending.done), "cudaEventSynchronize");
    detail::check(cudaEventDestroy(pending.done), "cudaEventDestroy");
    pending_.erase(it);

    const auto source_tier = offload ? StorageTier::GPU : StorageTier::CPU;
    const auto destination_tier = offload ? StorageTier::CPU : StorageTier::GPU;
    const auto& record = directory_.at(key);
    require(record.gpu.has_value() && record.cpu.has_value() &&
            record.cpu->location == (offload ? pending.destination : pending.source) &&
            record.gpu->location == (offload ? pending.source : pending.destination),
            "transfer replicas were modified while in flight");

    directory_.set_state(key, destination_tier, ChunkState::READY);
    directory_.set_state(key, source_tier, ChunkState::READY);
    directory_.unpin(key);
    if (release_source) {
        // 传输已完成且目录中不再有指向源副本的未决使用，释放安全。
        directory_.remove(key, source_tier);
    }
}

void GpuCpuTransfer::finish_offload(const CacheKey& key, bool release_gpu) {
    finish(key, true, release_gpu);
}

void GpuCpuTransfer::finish_onload(const CacheKey& key, bool release_cpu) {
    finish(key, false, release_cpu);
}

bool GpuCpuTransfer::has_pending(const CacheKey& key) const {
    return pending_.find(key) != pending_.end();
}

} // namespace kvflux::v3

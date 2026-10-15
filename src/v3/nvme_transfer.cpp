#include "kvflux/v3/nvme_transfer.h"

#include <stdexcept>

namespace kvflux::v3 {

NvmeTransfer::NvmeTransfer(CacheDirectory& directory, ChunkFileStore& store,
                           HostChunkBuffers& cpu_buffers)
    : directory_(directory), store_(store), cpu_buffers_(cpu_buffers) {}

void NvmeTransfer::offload(const CacheKey& key, std::uint64_t valid_tokens,
                           bool release_cpu) {
    const auto& record = directory_.at(key);
    if (!record.cpu.has_value() || record.cpu->state != ChunkState::READY) {
        throw std::logic_error("nvme offload requires a READY cpu replica");
    }
    if (record.nvme.has_value()) {
        throw std::logic_error("nvme replica already exists");
    }
    const auto source = record.cpu->location;
    directory_.set_state(key, StorageTier::CPU, ChunkState::COPYING);

    TierLocation destination{};
    try {
        destination = directory_.place(key, StorageTier::NVME, ChunkState::COPYING);
        const auto meta = make_file_meta(key, valid_tokens, cpu_buffers_.chunk_bytes());
        store_.write(destination.slot, meta, cpu_buffers_.data(source.slot));
    } catch (...) {
        if (directory_.at(key).nvme.has_value()) {
            directory_.set_state(key, StorageTier::NVME, ChunkState::ERROR);
            directory_.remove(key, StorageTier::NVME);
        }
        directory_.set_state(key, StorageTier::CPU, ChunkState::READY);
        throw;
    }
    directory_.set_state(key, StorageTier::NVME, ChunkState::READY);
    directory_.set_state(key, StorageTier::CPU, ChunkState::READY);
    if (release_cpu) directory_.remove(key, StorageTier::CPU);
}

void NvmeTransfer::onload(const CacheKey& key, bool release_nvme) {
    const auto& record = directory_.at(key);
    if (!record.nvme.has_value() || record.nvme->state != ChunkState::READY) {
        throw std::logic_error("nvme onload requires a READY nvme replica");
    }
    if (record.cpu.has_value()) {
        throw std::logic_error("cpu replica already exists");
    }
    const auto source = record.nvme->location;
    directory_.set_state(key, StorageTier::NVME, ChunkState::COPYING);

    try {
        const auto destination = directory_.place(key, StorageTier::CPU, ChunkState::LOADING);
        const auto expected = make_file_meta(key, 0, cpu_buffers_.chunk_bytes());
        const bool found = store_.read(source.slot, expected,
                                       cpu_buffers_.data(destination.slot),
                                       cpu_buffers_.chunk_bytes());
        if (!found) throw ChunkMissError("nvme chunk file missing for registered replica");
    } catch (...) {
        if (directory_.at(key).cpu.has_value()) {
            directory_.set_state(key, StorageTier::CPU, ChunkState::ERROR);
            directory_.remove(key, StorageTier::CPU);
        }
        directory_.set_state(key, StorageTier::NVME, ChunkState::READY);
        throw;
    }
    directory_.set_state(key, StorageTier::CPU, ChunkState::READY);
    directory_.set_state(key, StorageTier::NVME, ChunkState::READY);
    if (release_nvme) {
        store_.erase(source.slot);
        directory_.remove(key, StorageTier::NVME);
    }
}

} // namespace kvflux::v3

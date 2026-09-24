#include "kvflux/v2/paged_kv_storage.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace kvflux::v2 {
namespace {

std::size_t backing_pages(std::size_t physical_pages) {
    if (physical_pages > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::overflow_error("K/V backing page count overflow");
    }
    return 2 * physical_pages;
}

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        cuda_check(cudaGetDevice(&previous_), "cudaGetDevice");
        if (previous_ != device) cuda_check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceGuard() noexcept { (void)cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};

} // namespace

PagedKVStorage::PagedKVStorage(PhysicalBlockPool& pool, std::size_t num_kv_heads,
                               std::size_t head_size, DType dtype, int device)
    : pool_(pool),
      layout_(pool.capacity(), num_kv_heads, pool.block_size(), head_size, dtype),
      backing_(backing_pages(pool.capacity()), layout_.page_bytes(), pool.block_size(), device) {}

void* PagedKVStorage::key_base() const { return backing_.device_address(BlockId{0}); }

void* PagedKVStorage::value_base() const {
    return backing_.device_address(BlockId{pool_.capacity()});
}

void* PagedKVStorage::page_address(KVKind kind, PhysicalBlockHandle handle) const {
    const auto id = pool_.id(handle);
    const auto offset = layout_.page_byte_offset(kind, id);
    return static_cast<unsigned char*>(key_base()) + offset;
}

void* PagedKVStorage::element_address(KVKind kind, PhysicalBlockHandle handle,
                                      std::size_t head, std::size_t token,
                                      std::size_t dimension) const {
    const auto id = pool_.id(handle);
    const auto offset = layout_.byte_offset(kind, id, head, token, dimension);
    return static_cast<unsigned char*>(key_base()) + offset;
}

void* PagedKVStorage::slot_address(KVKind kind, PhysicalSlot slot,
                                   std::size_t head, std::size_t dimension) const {
    const auto offset = layout_.slot_byte_offset(kind, slot, head, dimension);
    return static_cast<unsigned char*>(key_base()) + offset;
}

void PagedKVStorage::copy_block(PhysicalBlockHandle source,
                                PhysicalBlockHandle destination) const {
    const auto source_id = pool_.id(source);
    const auto destination_id = pool_.id(destination);
    if (source_id == destination_id || pool_.ref_count(destination) != 1 ||
        pool_.is_published(destination)) {
        throw std::invalid_argument("copy destination must be a private, unpublished page");
    }
    DeviceGuard guard(device());
    try {
        // K/V 分别成区，两个 D2D 拷贝都要完成后才能切换请求的表项。
        cuda_check(cudaMemcpyAsync(page_address(KVKind::Key, destination),
                                   page_address(KVKind::Key, source), layout_.page_bytes(),
                                   cudaMemcpyDeviceToDevice), "copy K page");
        cuda_check(cudaMemcpyAsync(page_address(KVKind::Value, destination),
                                   page_address(KVKind::Value, source), layout_.page_bytes(),
                                   cudaMemcpyDeviceToDevice), "copy V page");
        cuda_check(cudaStreamSynchronize(nullptr), "synchronize KV page copy");
    } catch (...) {
        // 若 K 已入队而 V 拷贝失败，也先等待已入队工作，避免回滚后复用仍在写的页。
        (void)cudaStreamSynchronize(nullptr);
        throw;
    }
}

} // namespace kvflux::v2

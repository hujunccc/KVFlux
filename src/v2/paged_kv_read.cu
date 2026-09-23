#include "kvflux/v2/paged_kv_read.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace kvflux::v2 {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("paged KV read size overflow");
    }
    return a * b;
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

class DeviceBlocks {
public:
    explicit DeviceBlocks(std::size_t bytes) {
        cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc read block table");
    }
    ~DeviceBlocks() noexcept { (void)cudaFree(data_); }
    DeviceBlocks(const DeviceBlocks&) = delete;
    DeviceBlocks& operator=(const DeviceBlocks&) = delete;
    PhysicalBlockID* data() noexcept { return data_; }
private:
    PhysicalBlockID* data_ = nullptr;
};

void validate_device_pointer(const void* pointer, int device, const char* name) {
    cudaPointerAttributes attributes{};
    const auto result = cudaPointerGetAttributes(&attributes, pointer);
    if (result != cudaSuccess || attributes.type != cudaMemoryTypeDevice ||
        attributes.device != device) {
        if (result != cudaSuccess) (void)cudaGetLastError();
        throw std::invalid_argument(std::string(name) + " must be device memory on storage GPU");
    }
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto first = reinterpret_cast<std::uintptr_t>(a);
    const auto second = reinterpret_cast<std::uintptr_t>(b);
    return first <= second ? second - first < a_bytes : first - second < b_bytes;
}

template<class Element>
__global__ void paged_kv_read_kernel(const Element* key_cache, const Element* value_cache,
                                     const PhysicalBlockID* blocks, Element* key_out,
                                     Element* value_out, std::size_t total_elements,
                                     std::size_t heads, std::size_t block_size,
                                     std::size_t head_size) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const auto elements_per_token = heads * head_size;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_elements; index += stride) {
        const auto token = index / elements_per_token;
        const auto head = (index / head_size) % heads;
        const auto dimension = index % head_size;
        const auto physical_block = blocks[token / block_size];
        const auto token_offset = token % block_size;
        const auto cache_index = ((physical_block * heads + head) * block_size + token_offset)
                                 * head_size + dimension;
        key_out[index] = key_cache[cache_index];
        value_out[index] = value_cache[cache_index];
    }
}

template<class Element>
void launch_read(const PagedKVStorage& storage, const PhysicalBlockID* blocks,
                 void* key_out, void* value_out, std::size_t total_elements) {
    constexpr std::size_t threads = 256;
    const auto blocks_needed = total_elements / threads + (total_elements % threads != 0);
    const auto grid = static_cast<unsigned>(std::min<std::size_t>(blocks_needed, 1024));
    const auto& layout = storage.layout();
    paged_kv_read_kernel<<<grid, static_cast<unsigned>(threads)>>>(
        static_cast<const Element*>(storage.key_base()),
        static_cast<const Element*>(storage.value_base()), blocks,
        static_cast<Element*>(key_out), static_cast<Element*>(value_out),
        total_elements, layout.num_kv_heads(), layout.block_size(), layout.head_size());
    cuda_check(cudaGetLastError(), "paged KV read kernel launch");
}

} // namespace

void read_paged_kv(const PagedKVStorage& storage, const SequenceState& sequence,
                   void* key_out_device, void* value_out_device) {
    if (!storage.uses_pool(sequence.block_table())) {
        throw std::invalid_argument("sequence and KV storage must use one physical pool");
    }
    const auto& layout = storage.layout();
    if (sequence.block_size() != layout.block_size()) {
        throw std::invalid_argument("sequence and KV storage block sizes differ");
    }
    if (sequence.num_tokens() == 0) return;
    if (!key_out_device || !value_out_device) {
        throw std::invalid_argument("K and V output device pointers are required");
    }

    const auto host_blocks = build_read_block_table(sequence);
    for (auto block : host_blocks) {
        if (block >= layout.num_blocks()) throw std::out_of_range("KV read block out of range");
    }
    const auto total_elements = checked_multiply(
        checked_multiply(sequence.num_tokens(), layout.num_kv_heads()), layout.head_size());
    const auto output_bytes = checked_multiply(total_elements, layout.element_bytes());
    const auto table_bytes = checked_multiply(host_blocks.size(), sizeof(PhysicalBlockID));

    DeviceGuard guard(storage.device());
    validate_device_pointer(key_out_device, storage.device(), "K_out");
    validate_device_pointer(value_out_device, storage.device(), "V_out");
    if (overlaps(key_out_device, output_bytes, value_out_device, output_bytes) ||
        overlaps(key_out_device, output_bytes, storage.key_base(), layout.total_bytes()) ||
        overlaps(value_out_device, output_bytes, storage.key_base(), layout.total_bytes())) {
        throw std::invalid_argument("KV read outputs must not overlap each other or the cache");
    }

    DeviceBlocks device_blocks(table_bytes);
    cuda_check(cudaMemcpyAsync(device_blocks.data(), host_blocks.data(), table_bytes,
                               cudaMemcpyHostToDevice), "upload read block table");
    switch (layout.dtype()) {
    case DType::Float16:
    case DType::BFloat16:
        launch_read<std::uint16_t>(storage, device_blocks.data(), key_out_device,
                                   value_out_device, total_elements);
        break;
    case DType::Float32:
        launch_read<std::uint32_t>(storage, device_blocks.data(), key_out_device,
                                   value_out_device, total_elements);
        break;
    default:
        throw std::invalid_argument("unsupported paged KV read dtype");
    }
    cuda_check(cudaStreamSynchronize(nullptr), "paged KV read synchronize");
}

} // namespace kvflux::v2

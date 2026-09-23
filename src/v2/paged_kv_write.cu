#include "kvflux/v2/paged_kv_write.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace kvflux::v2 {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("paged KV write size overflow");
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

class DeviceSlots {
public:
    explicit DeviceSlots(std::size_t bytes) {
        cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc slot_mapping");
    }
    ~DeviceSlots() noexcept { (void)cudaFree(data_); }
    DeviceSlots(const DeviceSlots&) = delete;
    DeviceSlots& operator=(const DeviceSlots&) = delete;
    std::uint64_t* data() noexcept { return data_; }
private:
    std::uint64_t* data_ = nullptr;
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

template<class Element>
__global__ void paged_kv_write_kernel(const Element* key_new, const Element* value_new,
                                      const std::uint64_t* slots, Element* key_cache,
                                      Element* value_cache, std::size_t total_elements,
                                      std::size_t heads, std::size_t block_size,
                                      std::size_t head_size) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const auto elements_per_token = heads * head_size;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_elements; index += stride) {
        const auto batch_index = index / elements_per_token;
        const auto head = (index / head_size) % heads;
        const auto dimension = index % head_size;
        const auto slot = slots[batch_index];
        const auto physical_block = slot / block_size;
        const auto token_offset = slot % block_size;
        const auto cache_index = ((physical_block * heads + head) * block_size + token_offset)
                                 * head_size + dimension;
        key_cache[cache_index] = key_new[index];
        value_cache[cache_index] = value_new[index];
    }
}

template<class Element>
void launch_write(PagedKVStorage& storage, const void* key_new, const void* value_new,
                  const std::uint64_t* slots, std::size_t total_elements) {
    constexpr std::size_t threads = 256;
    const auto blocks_needed = total_elements / threads + (total_elements % threads != 0);
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(blocks_needed, 1024));
    const auto& layout = storage.layout();
    paged_kv_write_kernel<<<blocks, static_cast<unsigned>(threads)>>>(
        static_cast<const Element*>(key_new), static_cast<const Element*>(value_new),
        slots, static_cast<Element*>(storage.key_base()), static_cast<Element*>(storage.value_base()),
        total_elements, layout.num_kv_heads(), layout.block_size(), layout.head_size());
    cuda_check(cudaGetLastError(), "paged KV write kernel launch");
}

} // namespace

void write_paged_kv(PagedKVStorage& storage, const void* key_new_device,
                    const void* value_new_device, const SlotMapping& slot_mapping) {
    if (slot_mapping.empty()) return;
    if (!key_new_device || !value_new_device) {
        throw std::invalid_argument("K_new and V_new device pointers are required");
    }
    const auto& layout = storage.layout();
    std::unordered_set<PhysicalSlot> seen;
    seen.reserve(slot_mapping.size());
    for (auto slot : slot_mapping) {
        if (slot >= layout.slot_count()) throw std::out_of_range("KV write slot out of range");
        if (!seen.insert(slot).second) throw std::invalid_argument("duplicate KV write slot");
    }
    const auto total_elements = checked_multiply(
        checked_multiply(slot_mapping.size(), layout.num_kv_heads()), layout.head_size());
    (void)checked_multiply(total_elements, layout.element_bytes());
    const auto slot_bytes = checked_multiply(slot_mapping.size(), sizeof(PhysicalSlot));

    DeviceGuard guard(storage.device());
    validate_device_pointer(key_new_device, storage.device(), "K_new");
    validate_device_pointer(value_new_device, storage.device(), "V_new");
    DeviceSlots device_slots(slot_bytes);
    cuda_check(cudaMemcpyAsync(device_slots.data(), slot_mapping.data(), slot_bytes,
                               cudaMemcpyHostToDevice), "upload slot_mapping");
    switch (layout.dtype()) {
    case DType::Float16:
    case DType::BFloat16:
        launch_write<std::uint16_t>(storage, key_new_device, value_new_device,
                                    device_slots.data(), total_elements);
        break;
    case DType::Float32:
        launch_write<std::uint32_t>(storage, key_new_device, value_new_device,
                                    device_slots.data(), total_elements);
        break;
    default:
        throw std::invalid_argument("unsupported paged KV write dtype");
    }
    cuda_check(cudaStreamSynchronize(nullptr), "paged KV write synchronize");
}

} // namespace kvflux::v2

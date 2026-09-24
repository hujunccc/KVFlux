#include "kvflux/v2/paged_attention.h"

#include "kvflux/v2/paged_kv_read.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
        throw std::overflow_error("paged attention size overflow");
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
        cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc attention block table");
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
__global__ void paged_attention_kernel(const float* query, const Element* key_cache,
                                       const Element* value_cache, const PhysicalBlockID* blocks,
                                       float* output, std::size_t total_elements,
                                       std::size_t query_tokens, std::size_t kv_tokens,
                                       std::size_t query_heads, std::size_t kv_heads,
                                       std::size_t head_size, std::size_t block_size,
                                       double scale) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_elements; index += stride) {
        const auto dimension = index % head_size;
        const auto query_head = (index / head_size) % query_heads;
        const auto query_token = index / (query_heads * head_size);
        const auto kv_head = query_head / (query_heads / kv_heads);
        const auto query_base = index - dimension;
        const auto last_visible = kv_tokens - query_tokens + query_token;

        // 一个线程负责一个输出维度。逐 token 查页，重复计算 QK 点积以保持
        // 代码简单；整个 kernel 从不生成连续 K/V 中间缓冲。
        double maximum = -INFINITY;
        double denominator = 0.0;
        double numerator = 0.0;
        for (std::size_t token = 0; token <= last_visible; ++token) {
            const auto physical_block = blocks[token / block_size];
            const auto cache_base = ((physical_block * kv_heads + kv_head) * block_size +
                                     token % block_size) * head_size;
            double dot = 0.0;
            for (std::size_t dim = 0; dim < head_size; ++dim) {
                dot += static_cast<double>(query[query_base + dim]) *
                       static_cast<float>(key_cache[cache_base + dim]);
            }
            const double score = dot * scale;

            // 在线 softmax：当新 score 更大时，将旧分母和加权 V 重新缩放。
            // 相当于先对所有可见 score 减去最大值，再做 softmax @ V。
            const double next_maximum = score > maximum ? score : maximum;
            const double rescale = exp(maximum - next_maximum);
            const double weight = exp(score - next_maximum);
            denominator = denominator * rescale + weight;
            numerator = numerator * rescale + weight *
                        static_cast<float>(value_cache[cache_base + dimension]);
            maximum = next_maximum;
        }
        output[index] = static_cast<float>(numerator / denominator);
    }
}

template<class Element>
void launch_attention(const PagedKVStorage& storage, const PhysicalBlockID* blocks,
                      const float* query, float* output, std::size_t total_elements,
                      std::size_t query_tokens, std::size_t kv_tokens,
                      std::size_t query_heads) {
    constexpr std::size_t threads = 256;
    const auto needed = total_elements / threads + (total_elements % threads != 0);
    const auto grid = static_cast<unsigned>(std::min<std::size_t>(needed, 1024));
    const auto& layout = storage.layout();
    const double scale = 1.0 / std::sqrt(static_cast<double>(layout.head_size()));
    paged_attention_kernel<<<grid, static_cast<unsigned>(threads)>>>(
        query, static_cast<const Element*>(storage.key_base()),
        static_cast<const Element*>(storage.value_base()), blocks, output,
        total_elements, query_tokens, kv_tokens, query_heads, layout.num_kv_heads(),
        layout.head_size(), layout.block_size(), scale);
    cuda_check(cudaGetLastError(), "paged attention kernel launch");
}

} // namespace

void paged_attention(const PagedKVStorage& storage, const SequenceState& sequence,
                     const float* query_device, float* output_device,
                     std::size_t query_tokens, std::size_t query_heads) {
    if (!storage.uses_pool(sequence.block_table())) {
        throw std::invalid_argument("sequence and KV storage must use one physical pool");
    }
    const auto& layout = storage.layout();
    if (sequence.block_size() != layout.block_size() || query_heads == 0 ||
        query_heads % layout.num_kv_heads() != 0 || query_tokens > sequence.num_tokens()) {
        throw std::invalid_argument("invalid paged attention shape");
    }
    const auto total_elements = checked_multiply(
        checked_multiply(query_tokens, query_heads), layout.head_size());
    const auto output_bytes = checked_multiply(total_elements, sizeof(float));
    if (query_tokens == 0) return;
    if (!query_device || !output_device) {
        throw std::invalid_argument("Q and attention output device pointers are required");
    }

    const auto host_blocks = build_read_block_table(sequence);
    for (auto block : host_blocks) {
        if (block >= layout.num_blocks()) throw std::out_of_range("attention block out of range");
    }
    const auto table_bytes = checked_multiply(host_blocks.size(), sizeof(PhysicalBlockID));
    DeviceGuard guard(storage.device());
    validate_device_pointer(query_device, storage.device(), "Q");
    validate_device_pointer(output_device, storage.device(), "attention output");
    if (overlaps(query_device, output_bytes, output_device, output_bytes) ||
        overlaps(query_device, output_bytes, storage.key_base(), layout.total_bytes()) ||
        overlaps(output_device, output_bytes, storage.key_base(), layout.total_bytes())) {
        throw std::invalid_argument("Q and attention output must not overlap each other or KV cache");
    }

    DeviceBlocks device_blocks(table_bytes);
    cuda_check(cudaMemcpyAsync(device_blocks.data(), host_blocks.data(), table_bytes,
                               cudaMemcpyHostToDevice), "upload attention block table");
    switch (layout.dtype()) {
    case DType::Float16:
        launch_attention<__half>(storage, device_blocks.data(), query_device, output_device,
                                 total_elements, query_tokens, sequence.num_tokens(), query_heads);
        break;
    case DType::BFloat16:
        launch_attention<__nv_bfloat16>(storage, device_blocks.data(), query_device, output_device,
                                        total_elements, query_tokens, sequence.num_tokens(), query_heads);
        break;
    case DType::Float32:
        launch_attention<float>(storage, device_blocks.data(), query_device, output_device,
                                total_elements, query_tokens, sequence.num_tokens(), query_heads);
        break;
    default:
        throw std::invalid_argument("unsupported paged attention dtype");
    }
    cuda_check(cudaStreamSynchronize(nullptr), "paged attention synchronize");
}

} // namespace kvflux::v2

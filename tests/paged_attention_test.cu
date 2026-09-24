#include "kvflux/v2/paged_attention.h"
#include "kvflux/v2/paged_kv_read.h"
#include "kvflux/v2/paged_kv_write.h"
#include "kvflux/v2/reference_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Error, class Fn> void throws(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

template<class Element> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        CHECK(cudaMalloc(&data_, count * sizeof(Element)) == cudaSuccess);
    }
    ~DeviceBuffer() { (void)cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Element* data() const noexcept { return static_cast<Element*>(data_); }
private:
    void* data_ = nullptr;
};

template<class Element> Element encode(float x);
template<> float encode<float>(float x) { return x; }
template<> __half encode<__half>(float x) { return __float2half_rn(x); }
template<> __nv_bfloat16 encode<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

template<class Element> float decode(Element x);
template<> float decode<float>(float x) { return x; }
template<> float decode<__half>(__half x) { return __half2float(x); }
template<> float decode<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template<class Element>
float compare_paths(kvflux::DType dtype) {
    constexpr std::size_t tokens = 10, kv_heads = 2, query_heads = 4, head_size = 4;
    constexpr std::size_t kv_elements = tokens * kv_heads * head_size;
    kvflux::PhysicalBlockPool pool(8, 4);
    kvflux::v2::PagedKVStorage storage(pool, kv_heads, head_size, dtype);
    float worst_error = 0.0f;
    std::vector<kvflux::PhysicalBlockHandle> occupied;
    for (std::size_t i = 0; i < 5; ++i) occupied.push_back(pool.allocate());
    {
        kvflux::v2::SequenceState sequence(1001, pool);
        sequence.append_tokens(4);             // P5
        occupied.push_back(pool.allocate());    // P6
        pool.free(occupied[1]);                 // P1 可重用
        sequence.append_tokens(4);             // P1
        sequence.append_tokens(2);             // P7，只使用前两个 token
        CHECK((kvflux::v2::build_read_block_table(sequence) ==
               kvflux::v2::ReadBlockTable{5, 1, 7}));

        std::vector<kvflux::v2::TokenWrite> writes;
        for (std::size_t token = 0; token < tokens; ++token) writes.push_back({&sequence, token});
        const auto slots = kvflux::v2::build_slot_mapping(writes);
        std::vector<Element> key_input(kv_elements), value_input(kv_elements);
        for (std::size_t i = 0; i < kv_elements; ++i) {
            key_input[i] = encode<Element>(0.125f * static_cast<float>(static_cast<int>(i * 7 % 17) - 8));
            value_input[i] = encode<Element>(0.125f * static_cast<float>(static_cast<int>(i * 11 % 23) - 11));
        }
        DeviceBuffer<Element> key_device(kv_elements), value_device(kv_elements);
        DeviceBuffer<Element> gathered_key(kv_elements), gathered_value(kv_elements);
        CHECK(cudaMemset(storage.key_base(), 0xFF, storage.layout().total_bytes()) == cudaSuccess);
        CHECK(cudaMemcpy(key_device.data(), key_input.data(), kv_elements * sizeof(Element),
                         cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemcpy(value_device.data(), value_input.data(), kv_elements * sizeof(Element),
                         cudaMemcpyHostToDevice) == cudaSuccess);
        kvflux::v2::write_paged_kv(storage, key_device.data(), value_device.data(), slots);

        // Reference path：paged KV -> gather -> contiguous KV -> CPU attention。
        kvflux::v2::read_paged_kv(storage, sequence, gathered_key.data(), gathered_value.data());
        std::vector<Element> gathered_key_host(kv_elements), gathered_value_host(kv_elements);
        CHECK(cudaMemcpy(gathered_key_host.data(), gathered_key.data(), kv_elements * sizeof(Element),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(gathered_value_host.data(), gathered_value.data(), kv_elements * sizeof(Element),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        std::vector<float> contiguous_key(kv_elements), contiguous_value(kv_elements);
        for (std::size_t i = 0; i < kv_elements; ++i) {
            contiguous_key[i] = decode(gathered_key_host[i]);
            contiguous_value[i] = decode(gathered_value_host[i]);
            // Gather 结果还要和原始输入比较，避免两条路径共享同一种定位错误。
            CHECK(contiguous_key[i] == decode(key_input[i]));
            CHECK(contiguous_value[i] == decode(value_input[i]));
        }

        for (const auto query_tokens : {tokens, std::size_t{3}, std::size_t{1}}) {
            const auto query_elements = query_tokens * query_heads * head_size;
            std::vector<float> query(query_elements);
            for (std::size_t i = 0; i < query_elements; ++i) {
                query[i] = 0.125f * static_cast<float>(static_cast<int>(i * 5 % 19) - 9);
            }
            const auto expected = kvflux::v2::reference_attention(
                query, contiguous_key, contiguous_value,
                {query_tokens, tokens, query_heads, kv_heads, head_size});

            // KVFlux path：Q + block table + physical KV -> 直接 attention。
            DeviceBuffer<float> query_device(query_elements), output_device(query_elements + 1);
            CHECK(cudaMemcpy(query_device.data(), query.data(), query_elements * sizeof(float),
                             cudaMemcpyHostToDevice) == cudaSuccess);
            std::vector<float> actual(query_elements + 1, 12345.0f);
            CHECK(cudaMemcpy(output_device.data(), actual.data(), actual.size() * sizeof(float),
                             cudaMemcpyHostToDevice) == cudaSuccess);
            kvflux::v2::paged_attention(storage, sequence, query_device.data(), output_device.data(),
                                        query_tokens, query_heads);
            CHECK(cudaMemcpy(actual.data(), output_device.data(), actual.size() * sizeof(float),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
            float max_error = 0.0f;
            for (std::size_t i = 0; i < query_elements; ++i) {
                CHECK(std::isfinite(actual[i]));
                max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
            }
            CHECK(max_error < 1e-5f);
            worst_error = std::max(worst_error, max_error);
            CHECK(actual.back() == 12345.0f);

            if (query_tokens == 1) {
                throws<std::invalid_argument>([&] {
                    kvflux::v2::paged_attention(storage, sequence, nullptr, output_device.data(), 1, query_heads);
                });
                throws<std::invalid_argument>([&] {
                    kvflux::v2::paged_attention(storage, sequence, query_device.data(),
                                                query_device.data(), 1, query_heads);
                });
                throws<std::invalid_argument>([&] {
                    kvflux::v2::paged_attention(storage, sequence, query.data(),
                                                output_device.data(), 1, query_heads);
                });
                throws<std::invalid_argument>([&] {
                    kvflux::v2::paged_attention(storage, sequence, query_device.data(),
                                                output_device.data(), 1, 3);
                });
                throws<std::invalid_argument>([&] {
                    kvflux::v2::paged_attention(storage, sequence, query_device.data(),
                                                output_device.data(), tokens + 1, query_heads);
                });
            }
        }
        kvflux::v2::paged_attention(storage, sequence, nullptr, nullptr, 0, query_heads);
        kvflux::PhysicalBlockPool other_pool(8, 4);
        kvflux::v2::SequenceState other(1002, other_pool);
        other.append_tokens(1);
        throws<std::invalid_argument>([&] {
            kvflux::v2::paged_attention(storage, other, nullptr, nullptr, 0, query_heads);
        });
    }
    for (std::size_t i = 0; i < occupied.size(); ++i) {
        if (i != 1) pool.free(occupied[i]);
    }
    CHECK(pool.available() == pool.capacity());
    return worst_error;
}

int main() {
    try {
        int count = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && count == 0)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        const auto fp32_error = compare_paths<float>(kvflux::DType::Float32);
        const auto fp16_error = compare_paths<__half>(kvflux::DType::Float16);
        const auto bf16_error = compare_paths<__nv_bfloat16>(kvflux::DType::BFloat16);
        std::cout << "paged attention max_error (FP32/FP16/BF16): "
                  << fp32_error << " / " << fp16_error << " / " << bf16_error << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

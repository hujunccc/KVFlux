#include "kvflux/v2/batch_block_table.h"
#include "kvflux/v2/paged_attention.h"
#include "kvflux/v2/paged_kv_write.h"
#include "kvflux/v2/reference_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) { CHECK(cudaMalloc(&data_, count * sizeof(T)) == cudaSuccess); }
    ~DeviceBuffer() { (void)cudaFree(data_); }
    T* data() const noexcept { return static_cast<T*>(data_); }
private:
    void* data_ = nullptr;
};

template<class T> T encode(float x);
template<> float encode<float>(float x) { return x; }
template<> __half encode<__half>(float x) { return __float2half_rn(x); }
template<> __nv_bfloat16 encode<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template<class T> float decode(T x);
template<> float decode<float>(float x) { return x; }
template<> float decode<__half>(__half x) { return __half2float(x); }
template<> float decode<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template<class Element> void check_batch(kvflux::DType dtype) {
    constexpr std::size_t kv_heads = 2, query_heads = 4, head_size = 3;
    constexpr std::size_t tokens = 5 + 9 + 2;
    constexpr std::size_t kv_elements = tokens * kv_heads * head_size;
    constexpr std::size_t query_elements = 3 * query_heads * head_size;
    kvflux::PhysicalBlockPool pool(16, 4);
    kvflux::v2::PagedKVStorage storage(pool, kv_heads, head_size, dtype);
    kvflux::v2::SequenceState a(1, pool), b(2, pool), c(3, pool);
    a.append_tokens(4); // P0
    b.append_tokens(4); // P1
    c.append_tokens(2); // P2
    a.append_token();   // P3
    b.append_tokens(5); // P4, P5
    const std::vector<const kvflux::v2::SequenceState*> sequences{&a, &b, &c};
    const auto table = kvflux::v2::build_batch_block_table(sequences);
    CHECK((table.sequence_lengths() == std::vector<std::size_t>{5, 9, 2}));
    CHECK((table.block_table() == std::vector<kvflux::PhysicalBlockID>{
        0, 3, table.invalid_block(), 1, 4, 5, 2, table.invalid_block(), table.invalid_block()}));

    std::vector<kvflux::v2::TokenWrite> writes;
    for (const auto* sequence : sequences) {
        for (std::size_t token = 0; token < sequence->num_tokens(); ++token) {
            writes.push_back({sequence, token});
        }
    }
    const auto slots = kvflux::v2::build_slot_mapping(writes);
    std::vector<Element> keys(kv_elements), values(kv_elements);
    for (std::size_t i = 0; i < kv_elements; ++i) {
        keys[i] = encode<Element>(0.125f * static_cast<float>(static_cast<int>(i * 7 % 19) - 9));
        values[i] = encode<Element>(0.125f * static_cast<float>(static_cast<int>(i * 11 % 23) - 11));
    }
    DeviceBuffer<Element> key_device(kv_elements), value_device(kv_elements);
    CHECK(cudaMemset(storage.key_base(), 0xFF, storage.layout().total_bytes()) == cudaSuccess);
    CHECK(cudaMemcpy(key_device.data(), keys.data(), sizeof(Element) * kv_elements,
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(value_device.data(), values.data(), sizeof(Element) * kv_elements,
                     cudaMemcpyHostToDevice) == cudaSuccess);
    kvflux::v2::write_paged_kv(storage, key_device.data(), value_device.data(), slots);

    std::vector<float> query(query_elements);
    for (std::size_t i = 0; i < query_elements; ++i) {
        query[i] = 0.125f * static_cast<float>(static_cast<int>(i * 5 % 17) - 8);
    }
    DeviceBuffer<float> query_device(query_elements), output_device(query_elements + 1);
    CHECK(cudaMemcpy(query_device.data(), query.data(), sizeof(float) * query_elements,
                     cudaMemcpyHostToDevice) == cudaSuccess);
    std::vector<float> actual(query_elements + 1, 12345.0f);
    CHECK(cudaMemcpy(output_device.data(), actual.data(), sizeof(float) * actual.size(),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    kvflux::v2::paged_attention_batch(storage, sequences, query_device.data(),
                                      output_device.data(), query_heads);
    CHECK(cudaMemcpy(actual.data(), output_device.data(), sizeof(float) * actual.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

    std::size_t token_offset = 0;
    for (std::size_t request = 0; request < sequences.size(); ++request) {
        const auto length = sequences[request]->num_tokens();
        std::vector<float> q(query.begin() + request * query_heads * head_size,
                             query.begin() + (request + 1) * query_heads * head_size);
        std::vector<float> k(length * kv_heads * head_size), v(k.size());
        for (std::size_t i = 0; i < k.size(); ++i) {
            k[i] = decode(keys[token_offset * kv_heads * head_size + i]);
            v[i] = decode(values[token_offset * kv_heads * head_size + i]);
        }
        const auto expected = kvflux::v2::reference_attention(
            q, k, v, {1, length, query_heads, kv_heads, head_size});
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const float result = actual[request * expected.size() + i];
            CHECK(std::isfinite(result));
            CHECK(std::abs(result - expected[i]) < 1e-5f);
        }
        token_offset += length;
    }
    CHECK(actual.back() == 12345.0f);

    throws<std::invalid_argument>([&] {
        kvflux::v2::paged_attention_batch(storage, sequences, nullptr, output_device.data(), query_heads);
    });
    throws<std::invalid_argument>([&] {
        kvflux::v2::paged_attention_batch(storage, sequences, query_device.data(),
                                          query_device.data(), query_heads);
    });
    throws<std::invalid_argument>([&] {
        kvflux::v2::paged_attention_batch(storage, sequences, query.data(), output_device.data(), query_heads);
    });
    throws<std::invalid_argument>([&] {
        kvflux::v2::paged_attention_batch(storage, sequences, query_device.data(),
                                          output_device.data(), 3);
    });
    kvflux::v2::SequenceState empty(4, pool);
    throws<std::invalid_argument>([&] {
        kvflux::v2::paged_attention_batch(storage, {&a, &empty}, query_device.data(),
                                          output_device.data(), query_heads);
    });
    kvflux::PhysicalBlockPool other_pool(2, 4);
    kvflux::v2::SequenceState other(5, other_pool);
    other.append_token();
    throws<std::invalid_argument>([&] {
        kvflux::v2::paged_attention_batch(storage, {&a, &other}, query_device.data(),
                                          output_device.data(), query_heads);
    });
    kvflux::v2::paged_attention_batch(storage, {}, nullptr, nullptr, query_heads);
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
        check_batch<float>(kvflux::DType::Float32);
        check_batch<__half>(kvflux::DType::Float16);
        check_batch<__nv_bfloat16>(kvflux::DType::BFloat16);
        std::cout << "batch paged attention: 3 variable-length requests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

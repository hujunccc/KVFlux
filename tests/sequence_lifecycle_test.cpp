#include "kvflux/v2/reference_attention.h"
#include "kvflux/v2/sequence_lifecycle.h"

#include <cuda_runtime_api.h>

#include <cmath>
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

void upload(float* destination, const std::vector<float>& source) {
    CHECK(cudaMemcpy(destination, source.data(), source.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess);
}

void compare_attention(const float* device_output, const std::vector<float>& query,
                       const std::vector<float>& key, const std::vector<float>& value,
                       std::size_t query_tokens) {
    constexpr std::size_t query_heads = 2, kv_heads = 1, head_size = 2;
    const auto expected = kvflux::v2::reference_attention(
        query, key, value, {query_tokens, key.size() / (kv_heads * head_size),
                            query_heads, kv_heads, head_size});
    std::vector<float> actual(expected.size() + 1);
    CHECK(cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(actual[i]));
        CHECK(std::abs(actual[i] - expected[i]) < 1e-5f);
    }
    CHECK(actual.back() == 12345.0f); // attention 没有写出输出边界。
}

void full_request_lifecycle() {
    constexpr std::size_t query_heads = 2;
    kvflux::PhysicalBlockPool pool(4, 2);
    kvflux::v2::PagedKVStorage storage(pool, 1, 2, kvflux::DType::Float32);
    CHECK(cudaMemset(storage.key_base(), 0xFF, storage.layout().total_bytes()) == cudaSuccess);

    std::vector<float> keys{0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.5f};
    std::vector<float> values{1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f};
    const std::vector<float> prompt_query{0.1f, 0.2f, -0.3f, 0.4f,
                                          0.5f, -0.2f, 0.1f, 0.3f,
                                          -0.4f, 0.6f, 0.2f, -0.1f};
    const std::vector<float> fourth_k{0.6f, -0.3f}, fourth_v{2.0f, 1.5f};
    const std::vector<float> fourth_q{0.2f, 0.7f, -0.1f, 0.4f};
    const std::vector<float> fifth_k{-0.5f, 0.2f}, fifth_v{-2.0f, 3.0f};
    const std::vector<float> fifth_q{0.8f, -0.2f, 0.3f, 0.1f};

    DeviceBuffer<float> prompt_k_device(keys.size()), prompt_v_device(values.size());
    DeviceBuffer<float> prompt_q_device(prompt_query.size()), prompt_out_device(prompt_query.size() + 1);
    DeviceBuffer<float> one_k_device(2), one_v_device(2), one_q_device(4), one_out_device(5);
    upload(prompt_k_device.data(), keys);
    upload(prompt_v_device.data(), values);
    upload(prompt_q_device.data(), prompt_query);
    std::vector<float> prompt_sentinel(prompt_query.size() + 1, 12345.0f);
    std::vector<float> one_sentinel(5, 12345.0f);
    upload(prompt_out_device.data(), prompt_sentinel);

    {
        kvflux::v2::SequenceLifecycle request(1001, pool, storage, query_heads);
        CHECK(request.sequence().num_tokens() == 0 && pool.available() == 4);
        throws<std::invalid_argument>([&] {
            request.prefill(0, prompt_k_device.data(), prompt_v_device.data(),
                            prompt_q_device.data(), prompt_out_device.data());
        });
        CHECK(!request.finished());
        request.prefill(3, prompt_k_device.data(), prompt_v_device.data(),
                        prompt_q_device.data(), prompt_out_device.data());
        CHECK(request.sequence().num_tokens() == 3);
        CHECK(request.sequence().num_allocated_blocks() == 2 && pool.available() == 2);
        compare_attention(prompt_out_device.data(), prompt_query, keys, values, 3);
        throws<std::logic_error>([&] {
            request.prefill(3, prompt_k_device.data(), prompt_v_device.data(),
                            prompt_q_device.data(), prompt_out_device.data());
        });
        throws<std::invalid_argument>([&] {
            request.decode(nullptr, prompt_v_device.data(),
                           prompt_q_device.data(), one_out_device.data());
        });
        CHECK(!request.finished() && request.sequence().num_tokens() == 3);

        upload(one_k_device.data(), fourth_k);
        upload(one_v_device.data(), fourth_v);
        upload(one_q_device.data(), fourth_q);
        upload(one_out_device.data(), one_sentinel);
        const auto fourth = request.decode(one_k_device.data(), one_v_device.data(),
                                           one_q_device.data(), one_out_device.data());
        CHECK(fourth.offset_in_block == 1);
        CHECK(fourth.physical_block == request.sequence().physical_block_id(1));
        CHECK(request.sequence().num_tokens() == 4 && pool.available() == 2);
        keys.insert(keys.end(), fourth_k.begin(), fourth_k.end());
        values.insert(values.end(), fourth_v.begin(), fourth_v.end());
        compare_attention(one_out_device.data(), fourth_q, keys, values, 1);

        upload(one_k_device.data(), fifth_k);
        upload(one_v_device.data(), fifth_v);
        upload(one_q_device.data(), fifth_q);
        upload(one_out_device.data(), one_sentinel);
        const auto fifth = request.decode(one_k_device.data(), one_v_device.data(),
                                          one_q_device.data(), one_out_device.data());
        CHECK(fifth.offset_in_block == 0);
        CHECK(fifth.physical_block == request.sequence().physical_block_id(2));
        CHECK(fifth.physical_block != fourth.physical_block);
        CHECK(request.sequence().num_tokens() == 5 && pool.available() == 1);
        keys.insert(keys.end(), fifth_k.begin(), fifth_k.end());
        values.insert(values.end(), fifth_v.begin(), fifth_v.end());
        compare_attention(one_out_device.data(), fifth_q, keys, values, 1);

        request.finish();
        CHECK(request.finished() && pool.available() == pool.capacity());
        request.finish();
        throws<std::logic_error>([&] { (void)request.sequence(); });
        throws<std::logic_error>([&] {
            request.decode(one_k_device.data(), one_v_device.data(),
                           one_q_device.data(), one_out_device.data());
        });
    }
    CHECK(pool.available() == pool.capacity());

    // Attention 输入校验在页申请和 KV 写入之后失败：整个请求应被结束并归还页。
    kvflux::v2::SequenceLifecycle failed(1002, pool, storage, query_heads);
    throws<std::invalid_argument>([&] {
        failed.prefill(1, prompt_k_device.data(), prompt_v_device.data(),
                       prompt_query.data(), one_out_device.data());
    });
    CHECK(failed.finished() && pool.available() == pool.capacity());

    {
        kvflux::v2::SequenceLifecycle automatic(1003, pool, storage, query_heads);
        automatic.prefill(1, prompt_k_device.data(), prompt_v_device.data(),
                          prompt_q_device.data(), one_out_device.data());
        CHECK(pool.available() == pool.capacity() - 1);
    } // 未显式 finish，析构仍须归还最后一页。
    CHECK(pool.available() == pool.capacity());
}

void capacity_failure_releases_request() {
    kvflux::PhysicalBlockPool pool(1, 2);
    kvflux::v2::PagedKVStorage storage(pool, 1, 2, kvflux::DType::Float32);
    DeviceBuffer<float> kv_device(4), q_device(8), output_device(9);
    upload(kv_device.data(), {0.1f, 0.2f, 0.3f, 0.4f});
    upload(q_device.data(), {0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f, 0.8f, 0.7f});
    kvflux::v2::SequenceLifecycle request(2001, pool, storage, 2);
    request.prefill(2, kv_device.data(), kv_device.data(), q_device.data(), output_device.data());
    CHECK(pool.available() == 0);
    throws<kvflux::CapacityError>([&] {
        request.decode(kv_device.data(), kv_device.data(), q_device.data(), output_device.data());
    });
    CHECK(request.finished() && pool.available() == 1);
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
        full_request_lifecycle();
        capacity_failure_releases_request();
        std::cout << "sequence lifecycle: prefill, decode, attention and release passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

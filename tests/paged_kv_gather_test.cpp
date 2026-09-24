#include "kvflux/v2/paged_kv_read.h"
#include "kvflux/v2/paged_kv_write.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Element> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) { CHECK(cudaMalloc(&data_, count * sizeof(Element)) == cudaSuccess); }
    ~DeviceBuffer() { (void)cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Element* data() const noexcept { return static_cast<Element*>(data_); }
private:
    void* data_ = nullptr;
};

template<class Element>
void check_gather_and_isolation(kvflux::DType dtype) {
    constexpr std::size_t heads = 2, dimensions = 3, elements_per_token = heads * dimensions;
    kvflux::PhysicalBlockPool pool(5, 2);
    kvflux::v2::PagedKVStorage storage(pool, heads, dimensions, dtype);
    kvflux::v2::SequenceState a(101, pool), b(202, pool);

    // 交错分配使两个请求都使用不连续物理页，逻辑页号却各自从 0 开始。
    a.append_tokens(2); // A: P0
    b.append_tokens(2); // B: P1
    a.append_tokens(2); // A: P0, P2
    b.append_tokens(2); // B: P1, P3
    a.append_token();    // A: P0, P2, P4；最后一页只有一个 token。
    CHECK((kvflux::v2::build_read_block_table(a) == kvflux::v2::ReadBlockTable{0, 2, 4}));
    CHECK((kvflux::v2::build_read_block_table(b) == kvflux::v2::ReadBlockTable{1, 3}));

    // 写入 batch 顺序与任一请求的逻辑 token 顺序不同；预期结果直接按
    // (request, logical token, element) 生成，不依赖 slot mapping 的实现。
    const std::vector<kvflux::v2::TokenWrite> writes = {
        {&b, 0}, {&a, 0}, {&b, 1}, {&a, 1}, {&b, 2},
        {&a, 2}, {&b, 3}, {&a, 3}, {&a, 4},
    };
    std::vector<Element> input_k, input_v;
    std::vector<Element> expected_a_k(5 * elements_per_token), expected_a_v(expected_a_k.size());
    std::vector<Element> expected_b_k(4 * elements_per_token), expected_b_v(expected_b_k.size());
    for (const auto& write : writes) {
        const bool is_a = write.sequence == &a;
        auto& expected_k = is_a ? expected_a_k : expected_b_k;
        auto& expected_v = is_a ? expected_a_v : expected_b_v;
        for (std::size_t element = 0; element < elements_per_token; ++element) {
            const auto identity = (is_a ? 100u : 200u) +
                                  static_cast<unsigned>(write.token_position * elements_per_token + element);
            const auto key = static_cast<Element>(0x1000u + identity);
            const auto value = static_cast<Element>(0x3000u + identity);
            input_k.push_back(key);
            input_v.push_back(value);
            expected_k[write.token_position * elements_per_token + element] = key;
            expected_v[write.token_position * elements_per_token + element] = value;
        }
    }
    DeviceBuffer<Element> device_k(input_k.size()), device_v(input_v.size());
    CHECK(cudaMemcpy(device_k.data(), input_k.data(), input_k.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(device_v.data(), input_v.data(), input_v.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemset(storage.key_base(), 0xA5, storage.layout().cache_bytes()) == cudaSuccess);
    CHECK(cudaMemset(storage.value_base(), 0xA5, storage.layout().cache_bytes()) == cudaSuccess);
    kvflux::v2::write_paged_kv(storage, device_k.data(), device_v.data(),
                                kvflux::v2::build_slot_mapping(writes));

    const auto check_request = [&](const kvflux::v2::SequenceState& request,
                                   const std::vector<Element>& expected_k,
                                   const std::vector<Element>& expected_v) {
        DeviceBuffer<Element> output_k(expected_k.size() + 1), output_v(expected_v.size() + 1);
        CHECK(cudaMemset(output_k.data(), 0xA5, (expected_k.size() + 1) * sizeof(Element)) == cudaSuccess);
        CHECK(cudaMemset(output_v.data(), 0xA5, (expected_v.size() + 1) * sizeof(Element)) == cudaSuccess);
        kvflux::v2::read_paged_kv(storage, request, output_k.data(), output_v.data());
        std::vector<Element> actual_k(expected_k.size() + 1), actual_v(expected_v.size() + 1);
        CHECK(cudaMemcpy(actual_k.data(), output_k.data(), actual_k.size() * sizeof(Element),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(actual_v.data(), output_v.data(), actual_v.size() * sizeof(Element),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        for (std::size_t i = 0; i < expected_k.size(); ++i) {
            CHECK(actual_k[i] == expected_k[i]);
            CHECK(actual_v[i] == expected_v[i]);
        }
        const auto sentinel = static_cast<Element>(sizeof(Element) == 2 ? 0xA5A5u : 0xA5A5A5A5u);
        CHECK(actual_k.back() == sentinel && actual_v.back() == sentinel);
    };
    check_request(a, expected_a_k, expected_a_v);
    check_request(b, expected_b_k, expected_b_v);
    CHECK(pool.available() == 0);
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
        check_gather_and_isolation<std::uint16_t>(kvflux::DType::Float16);
        check_gather_and_isolation<std::uint16_t>(kvflux::DType::BFloat16);
        check_gather_and_isolation<std::uint32_t>(kvflux::DType::Float32);
        std::cout << "paged gather: interleaved requests match independent contiguous K/V\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

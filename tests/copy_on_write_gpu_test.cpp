#include "kvflux/v2/copy_on_write.h"
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

template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) { CHECK(cudaMalloc(&data_, count * sizeof(T)) == cudaSuccess); }
    ~DeviceBuffer() { (void)cudaFree(data_); }
    T* data() const noexcept { return static_cast<T*>(data_); }
private:
    void* data_ = nullptr;
};

template<class Element> void check_gpu_copy(kvflux::DType dtype) {
    constexpr std::size_t heads = 2, dimensions = 3, elements_per_token = heads * dimensions;
    kvflux::PhysicalBlockPool pool(3, 4);
    kvflux::v2::PagedKVStorage storage(pool, heads, dimensions, dtype);
    kvflux::v2::SequenceState a(1, pool);
    a.append_tokens(3);
    const auto old_page = a.physical_block_id(0);
    std::vector<Element> initial_k(3 * elements_per_token), initial_v(initial_k.size());
    for (std::size_t i = 0; i < initial_k.size(); ++i) {
        initial_k[i] = static_cast<Element>(0x1100 + i);
        initial_v[i] = static_cast<Element>(0x2200 + i);
    }
    DeviceBuffer<Element> initial_k_device(initial_k.size()), initial_v_device(initial_v.size());
    CHECK(cudaMemcpy(initial_k_device.data(), initial_k.data(), initial_k.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(initial_v_device.data(), initial_v.data(), initial_v.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    kvflux::v2::write_paged_kv(storage, initial_k_device.data(), initial_v_device.data(),
                                kvflux::v2::build_slot_mapping({{&a, 0}, {&a, 1}, {&a, 2}}));

    auto b = a.fork_shared(2);
    CHECK(pool.ref_count(a.block_table().handle(0)) == 2);
    const auto a_new = kvflux::v2::append_token_cow(storage, a);
    CHECK(a_new.physical_block != old_page && a_new.offset_in_block == 3);
    CHECK(a.num_tokens() == 4 && b.num_tokens() == 3);
    CHECK(b.physical_block_id(0) == old_page);

    std::vector<Element> a_k(elements_per_token), a_v(elements_per_token);
    std::vector<Element> b_k(elements_per_token), b_v(elements_per_token);
    for (std::size_t i = 0; i < elements_per_token; ++i) {
        a_k[i] = static_cast<Element>(0x3300 + i);
        a_v[i] = static_cast<Element>(0x4400 + i);
        b_k[i] = static_cast<Element>(0x5500 + i);
        b_v[i] = static_cast<Element>(0x6600 + i);
    }
    DeviceBuffer<Element> a_k_device(elements_per_token), a_v_device(elements_per_token);
    DeviceBuffer<Element> b_k_device(elements_per_token), b_v_device(elements_per_token);
    CHECK(cudaMemcpy(a_k_device.data(), a_k.data(), a_k.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(a_v_device.data(), a_v.data(), a_v.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(b_k_device.data(), b_k.data(), b_k.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(b_v_device.data(), b_v.data(), b_v.size() * sizeof(Element),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    kvflux::v2::write_paged_kv(storage, a_k_device.data(), a_v_device.data(),
                                kvflux::v2::build_slot_mapping({{&a, 3}}));

    DeviceBuffer<Element> a_read_k(4 * elements_per_token), a_read_v(4 * elements_per_token);
    DeviceBuffer<Element> b_read_k(4 * elements_per_token), b_read_v(4 * elements_per_token);
    kvflux::v2::read_paged_kv(storage, a, a_read_k.data(), a_read_v.data());
    kvflux::v2::read_paged_kv(storage, b, b_read_k.data(), b_read_v.data());
    std::vector<Element> actual_a_k(4 * elements_per_token), actual_a_v(actual_a_k.size());
    std::vector<Element> actual_b_k(3 * elements_per_token), actual_b_v(actual_b_k.size());
    CHECK(cudaMemcpy(actual_a_k.data(), a_read_k.data(), actual_a_k.size() * sizeof(Element),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(actual_a_v.data(), a_read_v.data(), actual_a_v.size() * sizeof(Element),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(actual_b_k.data(), b_read_k.data(), actual_b_k.size() * sizeof(Element),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(actual_b_v.data(), b_read_v.data(), actual_b_v.size() * sizeof(Element),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t i = 0; i < initial_k.size(); ++i) {
        CHECK(actual_a_k[i] == initial_k[i] && actual_b_k[i] == initial_k[i]);
        CHECK(actual_a_v[i] == initial_v[i] && actual_b_v[i] == initial_v[i]);
    }
    for (std::size_t i = 0; i < elements_per_token; ++i) {
        CHECK(actual_a_k[initial_k.size() + i] == a_k[i]);
        CHECK(actual_a_v[initial_v.size() + i] == a_v[i]);
    }

    const auto b_new = b.append_token(); // 旧页现在只有 B 引用，无需再复制。
    CHECK(b_new.physical_block == old_page && b_new.offset_in_block == 3);
    kvflux::v2::write_paged_kv(storage, b_k_device.data(), b_v_device.data(),
                                kvflux::v2::build_slot_mapping({{&b, 3}}));
    kvflux::v2::read_paged_kv(storage, b, b_read_k.data(), b_read_v.data());
    actual_b_k.resize(4 * elements_per_token);
    actual_b_v.resize(4 * elements_per_token);
    CHECK(cudaMemcpy(actual_b_k.data(), b_read_k.data(), actual_b_k.size() * sizeof(Element),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(actual_b_v.data(), b_read_v.data(), actual_b_v.size() * sizeof(Element),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t i = 0; i < elements_per_token; ++i) {
        CHECK(actual_b_k[initial_k.size() + i] == b_k[i]);
        CHECK(actual_b_v[initial_v.size() + i] == b_v[i]);
    }
    CHECK(a.physical_block_id(0) != b.physical_block_id(0));
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
        check_gpu_copy<std::uint16_t>(kvflux::DType::Float16);
        check_gpu_copy<std::uint16_t>(kvflux::DType::BFloat16);
        check_gpu_copy<std::uint32_t>(kvflux::DType::Float32);
        std::cout << "GPU copy-on-write: K/V preserved and requests isolated\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

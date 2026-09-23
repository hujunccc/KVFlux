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

template<class Element>
void check_paged_read(kvflux::DType dtype) {
    constexpr std::size_t heads = 2, dimensions = 3, tokens = 40;
    constexpr std::size_t elements = tokens * heads * dimensions;
    kvflux::PhysicalBlockPool pool(96, 16);
    kvflux::v2::PagedKVStorage storage(pool, heads, dimensions, dtype);
    std::vector<kvflux::PhysicalBlockHandle> occupied;
    for (std::size_t i = 0; i < 27; ++i) occupied.push_back(pool.allocate());
    {
        kvflux::v2::SequenceState request(1001, pool);
        request.append_tokens(16); // P27
        for (std::size_t i = 28; i < 91; ++i) occupied.push_back(pool.allocate());
        pool.free(occupied[3]);
        request.append_tokens(16); // P3
        request.append_tokens(8);  // P91
        CHECK((kvflux::v2::build_read_block_table(request) ==
               kvflux::v2::ReadBlockTable{27, 3, 91}));

        std::vector<kvflux::v2::TokenWrite> writes;
        for (std::size_t i = 0; i < tokens; ++i) writes.push_back({&request, i});
        const auto slots = kvflux::v2::build_slot_mapping(writes);
        std::vector<Element> keys(elements), values(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            keys[i] = static_cast<Element>(0x1000 + i);
            values[i] = static_cast<Element>(0x3000 + i);
        }

        DeviceBuffer<Element> key_input(elements), value_input(elements);
        DeviceBuffer<Element> key_output(elements + 1), value_output(elements + 1);
        CHECK(cudaMemcpy(key_input.data(), keys.data(), elements * sizeof(Element),
                         cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemcpy(value_input.data(), values.data(), elements * sizeof(Element),
                         cudaMemcpyHostToDevice) == cudaSuccess);
        kvflux::v2::write_paged_kv(storage, key_input.data(), value_input.data(), slots);

        CHECK(cudaMemset(key_output.data(), 0xA5, (elements + 1) * sizeof(Element)) == cudaSuccess);
        CHECK(cudaMemset(value_output.data(), 0xA5, (elements + 1) * sizeof(Element)) == cudaSuccess);
        kvflux::v2::read_paged_kv(storage, request, key_output.data(), value_output.data());

        std::vector<Element> actual_keys(elements + 1), actual_values(elements + 1);
        CHECK(cudaMemcpy(actual_keys.data(), key_output.data(), actual_keys.size() * sizeof(Element),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(actual_values.data(), value_output.data(), actual_values.size() * sizeof(Element),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        for (std::size_t i = 0; i < elements; ++i) {
            CHECK(actual_keys[i] == keys[i]);
            CHECK(actual_values[i] == values[i]);
        }
        const auto sentinel = static_cast<Element>(sizeof(Element) == 2 ? 0xA5A5u : 0xA5A5A5A5u);
        CHECK(actual_keys[elements] == sentinel);
        CHECK(actual_values[elements] == sentinel);

        throws<std::invalid_argument>([&] {
            kvflux::v2::read_paged_kv(storage, request, nullptr, value_output.data());
        });
        throws<std::invalid_argument>([&] {
            kvflux::v2::read_paged_kv(storage, request, key_output.data(), key_output.data());
        });
        throws<std::invalid_argument>([&] {
            kvflux::v2::read_paged_kv(storage, request, keys.data(), value_output.data());
        });
        throws<std::invalid_argument>([&] {
            kvflux::v2::read_paged_kv(storage, request, storage.key_base(), value_output.data());
        });

        kvflux::PhysicalBlockPool other_pool(96, 16);
        kvflux::v2::SequenceState other(1002, other_pool);
        other.append_tokens(1);
        throws<std::invalid_argument>([&] {
            kvflux::v2::read_paged_kv(storage, other, key_output.data(), value_output.data());
        });
        kvflux::v2::SequenceState empty(1003, pool);
        kvflux::v2::read_paged_kv(storage, empty, nullptr, nullptr);
    }
    for (std::size_t i = 0; i < occupied.size(); ++i) {
        if (i != 3) pool.free(occupied[i]);
    }
    CHECK(pool.available() == pool.capacity());
}

int main() {
    try {
        int count = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || (result == cudaSuccess && count == 0)) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        check_paged_read<std::uint16_t>(kvflux::DType::Float16);
        check_paged_read<std::uint16_t>(kvflux::DType::BFloat16);
        check_paged_read<std::uint32_t>(kvflux::DType::Float32);
        std::cout << "paged KV read: 40 tokens across [27, 3, 91] passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

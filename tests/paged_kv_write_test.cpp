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

template<class Element> class DeviceInput {
public:
    explicit DeviceInput(std::size_t count) {
        CHECK(cudaMalloc(&data_, count * sizeof(Element)) == cudaSuccess);
    }
    ~DeviceInput() { (void)cudaFree(data_); }
    DeviceInput(const DeviceInput&) = delete;
    DeviceInput& operator=(const DeviceInput&) = delete;
    Element* data() const noexcept { return static_cast<Element*>(data_); }
private:
    void* data_ = nullptr;
};

template<class Element>
void check_real_kernel_write(kvflux::DType dtype) {
    constexpr std::size_t heads = 2, dimensions = 4, block_size = 16;
    kvflux::PhysicalBlockPool pool(64, block_size);
    kvflux::v2::PagedKVStorage storage(pool, heads, dimensions, dtype);
    std::vector<kvflux::PhysicalBlockHandle> occupied;
    auto occupy_until_next_id = [&](std::size_t next_id) {
        while (pool.capacity() - pool.available() < next_id) occupied.push_back(pool.allocate());
    };

    {
        occupy_until_next_id(10);
        kvflux::v2::SequenceState a(1001, pool);
        a.append_tokens(14); // token 13 -> P10, slot 173。
        occupy_until_next_id(32);
        kvflux::v2::SequenceState b(1002, pool);
        b.append_tokens(11); // token 10 -> P32, slot 522。
        occupy_until_next_id(59);
        kvflux::v2::SequenceState c(1003, pool);
        c.append_tokens(1);  // token 0 -> P59, slot 944。

        const auto slots = kvflux::v2::build_slot_mapping({{&a, 13}, {&b, 10}, {&c, 0}});
        CHECK((slots == kvflux::v2::SlotMapping{173, 522, 944}));
        CHECK(cudaMemset(storage.key_base(), 0xA5, storage.layout().cache_bytes()) == cudaSuccess);
        CHECK(cudaMemset(storage.value_base(), 0xA5, storage.layout().cache_bytes()) == cudaSuccess);

        std::vector<Element> keys(slots.size() * heads * dimensions), values(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            keys[i] = static_cast<Element>(0x1200 + i);
            values[i] = static_cast<Element>(0x3400 + i);
        }
        DeviceInput<Element> key_device(keys.size()), value_device(values.size());
        CHECK(cudaMemcpy(key_device.data(), keys.data(), keys.size() * sizeof(Element),
                         cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemcpy(value_device.data(), values.data(), values.size() * sizeof(Element),
                         cudaMemcpyHostToDevice) == cudaSuccess);

        kvflux::v2::write_paged_kv(storage, key_device.data(), value_device.data(), slots);
        for (std::size_t batch = 0; batch < slots.size(); ++batch)
            for (std::size_t head = 0; head < heads; ++head)
                for (std::size_t dim = 0; dim < dimensions; ++dim) {
                    const auto input_index = (batch * heads + head) * dimensions + dim;
                    Element actual_key = 0, actual_value = 0;
                    CHECK(cudaMemcpy(&actual_key,
                                     storage.slot_address(kvflux::KVKind::Key, slots[batch], head, dim),
                                     sizeof(Element), cudaMemcpyDeviceToHost) == cudaSuccess);
                    CHECK(cudaMemcpy(&actual_value,
                                     storage.slot_address(kvflux::KVKind::Value, slots[batch], head, dim),
                                     sizeof(Element), cudaMemcpyDeviceToHost) == cudaSuccess);
                    CHECK(actual_key == keys[input_index] && actual_value == values[input_index]);
                }

        // 每个目标页的下一个 token 未在 batch 中，不能被 kernel 覆盖。
        const auto untouched = static_cast<Element>(sizeof(Element) == 2 ? 0xA5A5u : 0xA5A5A5A5u);
        for (auto slot : slots) {
            Element key_neighbor = 0, value_neighbor = 0;
            CHECK(cudaMemcpy(&key_neighbor, storage.slot_address(kvflux::KVKind::Key, slot + 1, 1, 3),
                             sizeof(Element), cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(cudaMemcpy(&value_neighbor, storage.slot_address(kvflux::KVKind::Value, slot + 1, 1, 3),
                             sizeof(Element), cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(key_neighbor == untouched && value_neighbor == untouched);
        }

        kvflux::v2::write_paged_kv(storage, nullptr, nullptr, {});
        throws<std::invalid_argument>([&] {
            kvflux::v2::write_paged_kv(storage, nullptr, value_device.data(), slots);
        });
        throws<std::out_of_range>([&] {
            kvflux::v2::write_paged_kv(storage, key_device.data(), value_device.data(),
                                       {storage.layout().slot_count()});
        });
        throws<std::invalid_argument>([&] {
            kvflux::v2::write_paged_kv(storage, key_device.data(), value_device.data(), {173, 173});
        });
        throws<std::invalid_argument>([&] {
            kvflux::v2::write_paged_kv(storage, keys.data(), value_device.data(), slots);
        });
    }
    for (auto handle : occupied) pool.free(handle);
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
        check_real_kernel_write<std::uint16_t>(kvflux::DType::Float16);
        check_real_kernel_write<std::uint16_t>(kvflux::DType::BFloat16);
        check_real_kernel_write<std::uint32_t>(kvflux::DType::Float32);
        std::cout << "paged KV write: 3-slot CUDA K/V roundtrip passed for FP16/BF16/FP32\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

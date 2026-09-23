#include "kvflux/v2/paged_kv_write.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
class DeviceInput {
public:
    explicit DeviceInput(std::size_t bytes) {
        if (cudaMalloc(&data_, bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc K_new/V_new failed");
        }
    }
    ~DeviceInput() { (void)cudaFree(data_); }
    DeviceInput(const DeviceInput&) = delete;
    DeviceInput& operator=(const DeviceInput&) = delete;
    void* data() const noexcept { return data_; }
private:
    void* data_ = nullptr;
};
}

int main() {
    try {
        kvflux::PhysicalBlockPool pool(64, 16);
        kvflux::v2::PagedKVStorage storage(pool, 1, 1, kvflux::DType::Float32);
        std::vector<kvflux::PhysicalBlockHandle> occupied;
        auto occupy_until_next_id = [&](std::size_t next_id) {
            while (pool.capacity() - pool.available() < next_id) occupied.push_back(pool.allocate());
        };

        {
            occupy_until_next_id(10);
            kvflux::v2::SequenceState a(1001, pool);
            a.append_tokens(14);
            occupy_until_next_id(32);
            kvflux::v2::SequenceState b(1002, pool);
            b.append_tokens(11);
            occupy_until_next_id(59);
            kvflux::v2::SequenceState c(1003, pool);
            c.append_tokens(1);
            const auto slots = kvflux::v2::build_slot_mapping({{&a, 13}, {&b, 10}, {&c, 0}});

            const float keys[] = {1.25f, 2.25f, 3.25f};
            const float values[] = {10.5f, 20.5f, 30.5f};
            DeviceInput key_device(sizeof(keys)), value_device(sizeof(values));
            if (cudaMemcpy(key_device.data(), keys, sizeof(keys), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(value_device.data(), values, sizeof(values), cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("upload K_new/V_new failed");
            }
            kvflux::v2::write_paged_kv(storage, key_device.data(), value_device.data(), slots);
            for (auto slot : slots) {
                float key = 0, value = 0;
                if (cudaMemcpy(&key, storage.slot_address(kvflux::KVKind::Key, slot, 0, 0),
                               sizeof(key), cudaMemcpyDeviceToHost) != cudaSuccess ||
                    cudaMemcpy(&value, storage.slot_address(kvflux::KVKind::Value, slot, 0, 0),
                               sizeof(value), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("read K/V cache failed");
                }
                std::cout << "slot " << slot << " -> K=" << key << " V=" << value << '\n';
            }
        }
        for (auto handle : occupied) pool.free(handle);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

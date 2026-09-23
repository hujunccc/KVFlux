#include "kvflux/v2/paged_kv_storage.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>

int main() {
    try {
        kvflux::PhysicalBlockPool pool(3, 2);
        kvflux::v2::PagedKVStorage storage(pool, 2, 3, kvflux::DType::Float32);
        kvflux::v2::SequenceState request(1001, pool);
        request.append_tokens(3); // token 2 开始使用物理页 1。
        const auto slot = kvflux::v2::build_slot_mapping({{&request, 2}}).at(0);

        const float key = 1.25f, value = 2.5f;
        auto* key_address = storage.slot_address(kvflux::KVKind::Key, slot, 1, 0);
        auto* value_address = storage.slot_address(kvflux::KVKind::Value, slot, 1, 0);
        if (cudaMemcpy(key_address, &key, sizeof(key), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(value_address, &value, sizeof(value), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("GPU K/V write failed");
        }
        float actual_key = 0, actual_value = 0;
        if (cudaMemcpy(&actual_key, key_address, sizeof(actual_key), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(&actual_value, value_address, sizeof(actual_value), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("GPU K/V read failed");
        }
        std::cout << "slot=" << slot << " K=" << actual_key << " V=" << actual_value
                  << " page_bytes=" << storage.layout().page_bytes() << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

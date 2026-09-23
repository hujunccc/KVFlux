#include "kvflux/v2/paged_kv_storage.h"

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

void real_gpu_storage_roundtrip() {
    using kvflux::KVKind;
    kvflux::PhysicalBlockPool pool(4, 2);
    kvflux::v2::PagedKVStorage storage(pool, 2, 3, kvflux::DType::Float32);
    const auto& layout = storage.layout();
    CHECK(layout.page_bytes() == 48 && layout.cache_bytes() == 192);
    CHECK(layout.total_bytes() == 384 && layout.slot_count() == 8);
    auto* key_base = static_cast<unsigned char*>(storage.key_base());
    auto* value_base = static_cast<unsigned char*>(storage.value_base());
    CHECK(value_base == key_base + layout.cache_bytes());

    cudaPointerAttributes key_attributes{}, value_attributes{};
    CHECK(cudaPointerGetAttributes(&key_attributes, key_base) == cudaSuccess);
    CHECK(cudaPointerGetAttributes(&value_attributes, value_base) == cudaSuccess);
    CHECK(key_attributes.type == cudaMemoryTypeDevice && value_attributes.type == cudaMemoryTypeDevice);
    CHECK(key_attributes.device == storage.device() && value_attributes.device == storage.device());

    kvflux::v2::SequenceState request(1001, pool);
    request.append_tokens(5); // P0、P1、P2；token 4 -> slot 4。
    const auto holder = pool.allocate(); // P3 用于带句柄的地址检查。
    CHECK(pool.id(holder) == 3);
    CHECK(storage.page_address(KVKind::Key, holder) == key_base + 3 * layout.page_bytes());
    CHECK(storage.page_address(KVKind::Value, holder) == value_base + 3 * layout.page_bytes());
    CHECK(storage.element_address(KVKind::Value, holder, 1, 1, 2) ==
          value_base + 3 * layout.page_bytes() + (1 * 2 * 3 + 1 * 3 + 2) * 4);
    CHECK(storage.slot_address(KVKind::Key, 6, 1, 1) ==
          storage.element_address(KVKind::Key, holder, 1, 0, 1));

    std::vector<std::uint32_t> keys(layout.cache_bytes() / 4), values(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = static_cast<std::uint32_t>(1000 + i);
        values[i] = static_cast<std::uint32_t>(10000 + i);
    }
    CHECK(cudaMemcpy(key_base, keys.data(), layout.cache_bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(value_base, values.data(), layout.cache_bytes(), cudaMemcpyHostToDevice) == cudaSuccess);

    const auto slots = kvflux::v2::build_slot_mapping({{&request, 0}, {&request, 2}, {&request, 4}});
    CHECK((slots == kvflux::v2::SlotMapping{0, 2, 4}));
    for (auto slot : slots)
        for (std::size_t head = 0; head < 2; ++head)
            for (std::size_t dim = 0; dim < 3; ++dim) {
                std::uint32_t actual_key = 0, actual_value = 0;
                CHECK(cudaMemcpy(&actual_key, storage.slot_address(KVKind::Key, slot, head, dim),
                                 sizeof(actual_key), cudaMemcpyDeviceToHost) == cudaSuccess);
                CHECK(cudaMemcpy(&actual_value, storage.slot_address(KVKind::Value, slot, head, dim),
                                 sizeof(actual_value), cudaMemcpyDeviceToHost) == cudaSuccess);
                const auto key_index = layout.slot_byte_offset(KVKind::Key, slot, head, dim) / 4;
                const auto value_index = (layout.slot_byte_offset(KVKind::Value, slot, head, dim)
                                          - layout.cache_bytes()) / 4;
                CHECK(actual_key == keys[key_index] && actual_value == values[value_index]);
            }

    throws<std::out_of_range>([&] { storage.slot_address(KVKind::Key, 8, 0, 0); });
    throws<std::out_of_range>([&] { storage.slot_address(KVKind::Key, 0, 2, 0); });
    throws<std::invalid_argument>([&] {
        storage.slot_address(static_cast<KVKind>(9), 0, 0, 0);
    });
    pool.free(holder);
    throws<std::invalid_argument>([&] { storage.page_address(KVKind::Key, holder); });
    const auto reused = pool.allocate();
    CHECK(reused.id == holder.id && reused.generation != holder.generation);
    CHECK(storage.page_address(KVKind::Key, reused) == key_base + 3 * layout.page_bytes());
    pool.free(reused);
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
        real_gpu_storage_roundtrip();
        std::cout << "paged KV storage: real GPU K/V slot roundtrip passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

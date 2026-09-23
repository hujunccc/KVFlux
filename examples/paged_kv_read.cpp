#include "kvflux/v2/paged_kv_read.h"
#include "kvflux/v2/paged_kv_write.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

} // namespace

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "CUDA device unavailable\n";
            return 0;
        }
        kvflux::PhysicalBlockPool pool(96, 16);
        kvflux::v2::PagedKVStorage storage(pool, 1, 1, kvflux::DType::Float32);
        std::vector<kvflux::PhysicalBlockHandle> occupied;
        for (std::size_t i = 0; i < 27; ++i) occupied.push_back(pool.allocate());
        {
            kvflux::v2::SequenceState request(1001, pool);
            request.append_tokens(16);
            for (std::size_t i = 28; i < 91; ++i) occupied.push_back(pool.allocate());
            pool.free(occupied[3]);
            request.append_tokens(16);
            request.append_tokens(8);

            std::vector<kvflux::v2::TokenWrite> writes;
            for (std::size_t i = 0; i < request.num_tokens(); ++i) writes.push_back({&request, i});
            const auto slots = kvflux::v2::build_slot_mapping(writes);
            std::vector<float> host_keys(40), host_values(40);
            for (std::size_t i = 0; i < 40; ++i) {
                host_keys[i] = static_cast<float>(i);
                host_values[i] = static_cast<float>(100 + i);
            }
            void *key_input = nullptr, *value_input = nullptr;
            void *key_output = nullptr, *value_output = nullptr;
            check(cudaMalloc(&key_input, 40 * sizeof(float)));
            check(cudaMalloc(&value_input, 40 * sizeof(float)));
            check(cudaMalloc(&key_output, 40 * sizeof(float)));
            check(cudaMalloc(&value_output, 40 * sizeof(float)));
            check(cudaMemcpy(key_input, host_keys.data(), 40 * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(value_input, host_values.data(), 40 * sizeof(float), cudaMemcpyHostToDevice));
            kvflux::v2::write_paged_kv(storage, key_input, value_input, slots);
            kvflux::v2::read_paged_kv(storage, request, key_output, value_output);
            check(cudaMemcpy(host_keys.data(), key_output, 40 * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(host_values.data(), value_output, 40 * sizeof(float), cudaMemcpyDeviceToHost));
            const auto blocks = kvflux::v2::build_read_block_table(request);
            std::cout << "Request " << request.request_id() << " blocks: ["
                      << blocks[0] << ", " << blocks[1] << ", " << blocks[2] << "]\n"
                      << "token 0: K=" << host_keys[0] << " V=" << host_values[0] << '\n'
                      << "token 16: K=" << host_keys[16] << " V=" << host_values[16] << '\n'
                      << "token 32: K=" << host_keys[32] << " V=" << host_values[32] << '\n'
                      << "token 39: K=" << host_keys[39] << " V=" << host_values[39] << '\n';
            check(cudaFree(key_input));
            check(cudaFree(value_input));
            check(cudaFree(key_output));
            check(cudaFree(value_output));
        }
        for (std::size_t i = 0; i < occupied.size(); ++i) {
            if (i != 3) pool.free(occupied[i]);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

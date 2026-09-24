#include "kvflux/v2/sequence_lifecycle.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceFloats {
public:
    explicit DeviceFloats(std::size_t count) {
        cuda_check(cudaMalloc(&data_, count * sizeof(float)), "cudaMalloc");
    }
    ~DeviceFloats() { (void)cudaFree(data_); }
    float* data() const noexcept { return static_cast<float*>(data_); }
    void upload(const std::vector<float>& values) {
        cuda_check(cudaMemcpy(data_, values.data(), values.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
private:
    void* data_ = nullptr;
};

} // namespace

int main() {
    try {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) {
            std::cout << "No usable CUDA device\n";
            return 0;
        }
        cuda_check(status, "cudaGetDeviceCount");

        kvflux::PhysicalBlockPool pool(2, 2);
        kvflux::v2::PagedKVStorage storage(pool, 1, 2, kvflux::DType::Float32);
        DeviceFloats prompt_k(4), prompt_v(4), prompt_q(4), prompt_o(4);
        DeviceFloats next_k(2), next_v(2), next_q(2), next_o(2);
        prompt_k.upload({0.1f, 0.2f, 0.3f, 0.4f});
        prompt_v.upload({1.0f, 2.0f, 3.0f, 4.0f});
        prompt_q.upload({0.2f, 0.1f, 0.4f, 0.3f});
        next_k.upload({0.5f, 0.6f});
        next_v.upload({5.0f, 6.0f});
        next_q.upload({0.1f, 0.7f});

        kvflux::v2::SequenceLifecycle request(1001, pool, storage, 1);
        request.prefill(2, prompt_k.data(), prompt_v.data(), prompt_q.data(), prompt_o.data());
        const auto location = request.decode(next_k.data(), next_v.data(),
                                             next_q.data(), next_o.data());
        float result[2]{};
        cuda_check(cudaMemcpy(result, next_o.data(), sizeof(result), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
        std::cout << "decode token: P" << location.physical_block << "+" << location.offset_in_block
                  << ", attention=[" << result[0] << ", " << result[1] << "]\n";
        request.finish();
        std::cout << "reclaimable pages: " << pool.available() << '/' << pool.capacity() << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

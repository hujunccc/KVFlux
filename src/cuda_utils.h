#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace kvflux::detail {
inline void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous_), "cudaGetDevice");
        if (device != previous_) check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceGuard() noexcept { (void)cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};
} // namespace kvflux::detail

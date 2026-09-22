#pragma once
#include <cuda_runtime_api.h>

namespace kvflux {
// 独立 non-blocking stream，不与 legacy default stream 隐式串行化。
class CudaStream {
public:
    explicit CudaStream(int device = 0);
    ~CudaStream() noexcept;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    void synchronize() const;
    cudaStream_t native_handle() const noexcept { return stream_; }
    int device() const noexcept { return device_; }
private:
    int device_;
    cudaStream_t stream_ = nullptr;
};
} // namespace kvflux

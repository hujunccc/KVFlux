#include "kvflux/cuda_stream.h"
#include "cuda_utils.h"

namespace kvflux {
CudaStream::CudaStream(int device) : device_(device) {
    int count = 0;
    detail::check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (device < 0 || device >= count) throw std::invalid_argument("stream device out of range");
    detail::DeviceGuard guard(device_);
    detail::check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}
CudaStream::~CudaStream() noexcept {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device_) == cudaSuccess) {
        (void)cudaStreamSynchronize(stream_);
        (void)cudaStreamDestroy(stream_);
        (void)cudaSetDevice(previous);
    }
}
void CudaStream::synchronize() const {
    detail::DeviceGuard guard(device_);
    detail::check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}
} // namespace kvflux

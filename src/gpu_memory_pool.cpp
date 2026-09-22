#include "kvflux/gpu_memory_pool.h"

#include <cuda_runtime_api.h>
#include <limits>
#include <string>

namespace kvflux {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

std::size_t checked_pool_bytes(std::size_t blocks, std::size_t bytes) {
    if (blocks == 0 || bytes == 0) throw std::invalid_argument("pool dimensions must be positive");
    if (blocks > std::numeric_limits<std::size_t>::max() / bytes) {
        throw std::overflow_error("GPU pool byte size overflow");
    }
    return blocks * bytes;
}

// 每次 CUDA 操作临时切到池所属设备，退出时恢复调用者原来的设备。
class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        cuda_check(cudaGetDevice(&previous_), "cudaGetDevice");
        if (device != previous_) cuda_check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceGuard() noexcept { (void)cudaSetDevice(previous_); }
private:
    int previous_ = 0;
};

} // namespace

GpuMemoryPool::GpuMemoryPool(std::size_t blocks, std::size_t bytes,
                             std::size_t tokens_per_block, int device)
    : manager_(checked_pool_bytes(blocks, bytes) / bytes, tokens_per_block),
      block_bytes_(bytes), allocated_bytes_(checked_pool_bytes(blocks, bytes)), device_(device),
      initialized_(blocks, false) {
    if (device < 0) throw std::invalid_argument("device must be nonnegative");
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device >= device_count) throw std::invalid_argument("GPU device index out of range");
    DeviceGuard guard(device_);
    // 唯一一次设备内存分配。成功之后没有可能抛异常的初始化步骤。
    cuda_check(cudaMalloc(&base_, allocated_bytes_), "cudaMalloc KV pool");
}

GpuMemoryPool::GpuMemoryPool(const KVBlockLayout& layout, std::size_t budget_bytes, int device)
    : GpuMemoryPool(layout.capacity_for(budget_bytes).total_blocks, layout.block_bytes(),
                    layout.config().block_size, device) {}

GpuMemoryPool::~GpuMemoryPool() noexcept {
    // 析构不能抛异常；若 CUDA 上下文已被外部重置，则无法保证释放成功。
    // 因此池必须在 cudaDeviceReset 之前销毁。
    int previous = 0;
    if (base_ && cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device_) == cudaSuccess) {
        (void)cudaFree(base_);
        (void)cudaSetDevice(previous);
    }
}

std::size_t GpuMemoryPool::remaining_blocks() const noexcept {
    const auto s = manager_.stats();
    return s.free + s.cached_idle;
}

BlockHandle GpuMemoryPool::allocate() {
    auto h = manager_.allocate();
    // 槽位地址可复用，但旧 generation 的数据不能当作新块的有效内容。
    initialized_[h.id] = false;
    return h;
}

void* GpuMemoryPool::device_address(BlockId id) const {
    if (id >= total_blocks()) throw std::out_of_range("GPU block id out of range");
    return static_cast<unsigned char*>(base_) + id * block_bytes_;
}

void* GpuMemoryPool::device_address(BlockHandle h) const {
    if (manager_.ref_count(h) == 0) throw std::invalid_argument("GPU block has no active reference");
    return device_address(h.id);
}

void GpuMemoryPool::write_block(BlockHandle h, const void* host, std::size_t bytes) {
    auto* destination = device_address(h);
    if (!host || bytes != block_bytes_) throw std::invalid_argument("write requires one full host block");
    if (manager_.ref_count(h) != 1 || manager_.is_published(h)) {
        throw std::invalid_argument("cannot overwrite a shared or published block");
    }
    DeviceGuard guard(device_);
    initialized_[h.id] = false; // 传输失败时，不能继续把旧内容当作有效块发布。
    cuda_check(cudaMemcpy(destination, host, bytes, cudaMemcpyHostToDevice), "write_block cudaMemcpy");
    // pageable host 内存的 H2D 返回不一定代表设备端传输已结束，显式等待。
    cuda_check(cudaStreamSynchronize(nullptr), "write_block synchronize");
    initialized_[h.id] = true;
}

void GpuMemoryPool::read_block(BlockHandle h, void* host, std::size_t bytes) const {
    const auto* source = device_address(h);
    if (!host || bytes != block_bytes_) throw std::invalid_argument("read requires one full host block");
    if (!initialized_[h.id]) throw std::invalid_argument("block has not been written");
    DeviceGuard guard(device_);
    cuda_check(cudaMemcpy(host, source, bytes, cudaMemcpyDeviceToHost), "read_block cudaMemcpy");
}

void GpuMemoryPool::publish(BlockHandle h, const Tokens& prefix) {
    (void)device_address(h);
    if (!initialized_[h.id]) throw std::invalid_argument("cannot publish unwritten GPU block");
    manager_.publish(h, prefix);
}

} // namespace kvflux

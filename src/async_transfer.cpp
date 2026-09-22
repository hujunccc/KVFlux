#include "kvflux/async_transfer.h"
#include "pinned_storage.h"
#include <unordered_set>

namespace kvflux {
struct AsyncTransferRuntime::Pending {
    std::vector<BlockHandle> blocks;
    std::shared_ptr<PinnedBuffer::Storage> buffer;
    bool write = false;
    int device;
    cudaEvent_t begin = nullptr, end = nullptr;
    explicit Pending(int device_id) : device(device_id) {
        detail::DeviceGuard guard(device);
        detail::check(cudaEventCreate(&begin), "transfer begin event");
        try { detail::check(cudaEventCreate(&end), "transfer end event"); }
        catch (...) { (void)cudaEventDestroy(begin); throw; }
    }
    ~Pending() noexcept {
        int previous = 0;
        if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
            (void)cudaEventDestroy(begin);
            (void)cudaEventDestroy(end);
            (void)cudaSetDevice(previous);
        }
    }
};

AsyncTransferRuntime::AsyncTransferRuntime(GpuMemoryPool& pool)
    : pool_(pool), compute_(pool.device()), transfer_(pool.device()) {}

AsyncTransferRuntime::~AsyncTransferRuntime() noexcept {
    // 即使调用方忘记 synchronize，也等待 DMA 后再释放 pinned memory 和块引用。
    try { compute_.synchronize(); } catch (...) {}
    try { synchronize(); } catch (...) { /* 显式 synchronize 才能向调用者报告设备错误。 */ }
}

void AsyncTransferRuntime::write_batch(const std::vector<BlockHandle>& blocks, PinnedBuffer& source) {
    submit(blocks, source, true);
}
void AsyncTransferRuntime::read_batch(const std::vector<BlockHandle>& blocks, PinnedBuffer& destination) {
    submit(blocks, destination, false);
}

void AsyncTransferRuntime::submit(const std::vector<BlockHandle>& blocks, PinnedBuffer& buffer, bool write) {
    if (blocks.empty()) throw std::invalid_argument("empty transfer batch");
    // 使用除法和余数验证长度，避免 blocks.size()*block_bytes 的溢出。
    if (buffer.size() / pool_.block_bytes() != blocks.size() || buffer.size() % pool_.block_bytes()) {
        throw std::invalid_argument("batch buffer must contain exactly N full blocks");
    }
    if (buffer.busy()) throw std::logic_error("pinned buffer is already in flight");
    std::unordered_set<BlockId> seen;
    for (auto h : blocks) {
        (void)pool_.device_address(h);
        if (!seen.insert(h.id).second) throw std::invalid_argument("duplicate block in batch");
        if (pool_.in_flight_[h.id]) throw std::logic_error("block is already in flight");
        if (write && (pool_.ref_count(h) != 1 || pool_.manager_.is_published(h))) {
            throw std::invalid_argument("async write requires an exclusive unpublished block");
        }
        if (!write && !pool_.initialized_[h.id]) throw std::invalid_argument("block has not been written");
    }
    auto work = recycled_.empty() ? std::make_unique<Pending>(pool_.device()) : std::move(recycled_.back());
    if (!recycled_.empty()) recycled_.pop_back();
    work->blocks = blocks;
    work->buffer = buffer.storage_;
    work->write = write;
    pending_.reserve(pending_.size() + 1);
    recycled_.reserve(recycled_.size() + pending_.size() + 1); // finish() 必须不分配内存。
    detail::DeviceGuard guard(pool_.device());
    std::size_t retained = 0;
    try {
        for (auto h : blocks) { pool_.retain(h); ++retained; }
    } catch (...) {
        for (std::size_t i = 0; i < retained; ++i) pool_.release(blocks[i]);
        throw;
    }
    buffer.storage_->busy = true;
    for (auto h : blocks) {
        pool_.in_flight_[h.id] = true;
        if (write) pool_.initialized_[h.id] = false;
    }
    pending_.push_back(std::move(work)); // 已 reserve；从此异常路径也能统一清理。
    try {
        detail::check(cudaEventRecord(pending_.back()->begin, transfer_.native_handle()), "record transfer begin");
        auto* host = static_cast<unsigned char*>(buffer.storage_->pointer);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            auto* gpu = pool_.device_address(blocks[i]);
            auto* cpu = host + i * pool_.block_bytes();
            detail::check(cudaMemcpyAsync(write ? gpu : cpu, write ? cpu : gpu,
                          pool_.block_bytes(), write ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                          transfer_.native_handle()), "batch cudaMemcpyAsync");
        }
        detail::check(cudaEventRecord(pending_.back()->end, transfer_.native_handle()), "record transfer end");
    } catch (...) {
        // 部分提交失败时也必须等已提交的 DMA 结束，不能提前销毁 buffer。
        (void)cudaStreamSynchronize(transfer_.native_handle());
        finish(false);
        throw;
    }
}

void AsyncTransferRuntime::finish(bool success) noexcept {
    if (!success) metrics_.failed_batches += pending_.size();
    for (auto& work : pending_) {
        for (auto h : work->blocks) {
            if (work->write) pool_.initialized_[h.id] = success;
            pool_.in_flight_[h.id] = false;
            pool_.release(h); // 归还运行时持有的引用；调用者可能已经释放自己的引用。
        }
        work->buffer->busy = false;
        work->buffer.reset();
        work->blocks.clear();
        recycled_.push_back(std::move(work));
    }
    pending_.clear();
}

void AsyncTransferRuntime::synchronize() {
    try {
        detail::DeviceGuard guard(pool_.device());
        transfer_.synchronize();
        auto updated = metrics_;
        for (const auto& work : pending_) {
            float ms = 0;
            detail::check(cudaEventElapsedTime(&ms, work->begin, work->end), "transfer elapsed time");
            auto& direction = work->write ? updated.cpu_to_gpu : updated.gpu_to_cpu;
            direction.bytes += work->buffer->bytes;
            ++direction.batches;
            direction.device_ms += ms;
        }
        metrics_ = updated;
    }
    catch (...) { finish(false); throw; }
    finish(true);
}

void AsyncTransferRuntime::compute_wait_for_transfers() {
    detail::DeviceGuard guard(pool_.device());
    cudaEvent_t event = nullptr;
    detail::check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
    try {
        detail::check(cudaEventRecord(event, transfer_.native_handle()), "cudaEventRecord");
        detail::check(cudaStreamWaitEvent(compute_.native_handle(), event, 0), "cudaStreamWaitEvent");
    } catch (...) { (void)cudaEventDestroy(event); throw; }
    // CUDA 允许销毁尚未完成的 event，相关资源在设备完成使用后才释放。
    detail::check(cudaEventDestroy(event), "cudaEventDestroy");
}
} // namespace kvflux

#pragma once
#include <cstddef>
#include <memory>

namespace kvflux {
class AsyncTransferRuntime;

// 长期复用的 page-locked host buffer。分配不放在每次传输的热路径中。
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes);
    ~PinnedBuffer();
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&&) = delete;
    PinnedBuffer& operator=(PinnedBuffer&&) = delete;
    std::size_t size() const noexcept;
    bool busy() const noexcept;
    // 在途传输期间拒绝获取指针；此前取得的裸指针也不得访问或修改。
    void* data();
    const void* data() const;
    void copy_from(const void* source, std::size_t bytes, std::size_t offset = 0);
    void copy_to(void* destination, std::size_t bytes, std::size_t offset = 0) const;
private:
    struct Storage;
    std::shared_ptr<Storage> storage_;
    friend class AsyncTransferRuntime;
};
} // namespace kvflux

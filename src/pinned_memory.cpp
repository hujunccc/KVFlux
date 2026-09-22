#include "pinned_storage.h"
#include <cstring>

namespace kvflux {
PinnedBuffer::PinnedBuffer(std::size_t bytes) : storage_(std::make_shared<Storage>(bytes)) {}
PinnedBuffer::~PinnedBuffer() = default;
std::size_t PinnedBuffer::size() const noexcept { return storage_->bytes; }
bool PinnedBuffer::busy() const noexcept { return storage_->busy; }
const void* PinnedBuffer::data() const {
    if (busy()) throw std::logic_error("pinned buffer is in flight; synchronize before host access");
    return storage_->pointer;
}
void* PinnedBuffer::data() { return const_cast<void*>(static_cast<const PinnedBuffer&>(*this).data()); }
void PinnedBuffer::copy_from(const void* source, std::size_t bytes, std::size_t offset) {
    if (!source || offset > size() || bytes > size() - offset) throw std::invalid_argument("invalid staging source/range");
    std::memmove(static_cast<unsigned char*>(data()) + offset, source, bytes);
}
void PinnedBuffer::copy_to(void* destination, std::size_t bytes, std::size_t offset) const {
    if (!destination || offset > size() || bytes > size() - offset) throw std::invalid_argument("invalid staging destination/range");
    std::memmove(destination, static_cast<const unsigned char*>(data()) + offset, bytes);
}
} // namespace kvflux

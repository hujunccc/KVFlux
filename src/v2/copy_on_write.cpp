#include "kvflux/v2/copy_on_write.h"

#include <stdexcept>

namespace kvflux::v2 {

TokenLocation append_token_cow(const PagedKVStorage& storage, SequenceState& sequence) {
    if (!storage.uses_pool(sequence.block_table()) ||
        storage.layout().block_size() != sequence.block_size()) {
        throw std::invalid_argument("sequence and KV storage must use one physical pool");
    }
    return sequence.append_token_with_copy([&](PhysicalBlockHandle source,
                                                PhysicalBlockHandle destination) {
        storage.copy_block(source, destination);
    });
}

} // namespace kvflux::v2

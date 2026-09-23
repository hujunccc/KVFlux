#include "kvflux/v2/paged_kv_read.h"

namespace kvflux::v2 {

ReadBlockTable build_read_block_table(const SequenceState& sequence) {
    ReadBlockTable blocks;
    blocks.reserve(sequence.num_allocated_blocks());
    for (std::size_t logical = 0; logical < sequence.num_allocated_blocks(); ++logical) {
        blocks.push_back(sequence.physical_block_id(logical));
    }
    return blocks;
}

} // namespace kvflux::v2

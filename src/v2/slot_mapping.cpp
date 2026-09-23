#include "kvflux/v2/slot_mapping.h"

#include <limits>
#include <stdexcept>

namespace kvflux::v2 {

SlotMapping build_slot_mapping(const std::vector<TokenWrite>& batch) {
    static_assert(sizeof(std::size_t) <= sizeof(PhysicalSlot),
                  "slot mapping requires size_t to fit in uint64_t");
    SlotMapping slots;
    slots.reserve(batch.size());
    if (batch.empty()) return slots;
    if (!batch.front().sequence) throw std::invalid_argument("null sequence in slot mapping");

    const auto& first_table = batch.front().sequence->block_table();
    for (const auto& write : batch) {
        if (!write.sequence) throw std::invalid_argument("null sequence in slot mapping");
        if (!first_table.shares_pool_with(write.sequence->block_table())) {
            throw std::invalid_argument("slot mapping batch must use one physical pool");
        }
        const auto location = write.sequence->token_location(write.token_position);
        const auto block_size = static_cast<PhysicalSlot>(write.sequence->block_size());
        const auto block = static_cast<PhysicalSlot>(location.physical_block);
        const auto offset = static_cast<PhysicalSlot>(location.offset_in_block);
        if (block > (std::numeric_limits<PhysicalSlot>::max() - offset) / block_size) {
            throw std::overflow_error("physical slot overflow");
        }
        slots.push_back(block * block_size + offset);
    }
    return slots;
}

} // namespace kvflux::v2

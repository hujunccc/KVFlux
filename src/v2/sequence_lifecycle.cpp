#include "kvflux/v2/sequence_lifecycle.h"

#include "kvflux/v2/copy_on_write.h"
#include "kvflux/v2/paged_attention.h"
#include "kvflux/v2/paged_kv_write.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <vector>

namespace kvflux::v2 {

SequenceLifecycle::SequenceLifecycle(RequestID request_id, PhysicalBlockPool& pool,
                                     PagedKVStorage& storage, std::size_t query_heads)
    : storage_(storage), sequence_(std::in_place, request_id, pool), query_heads_(query_heads) {
    if (!storage_.uses_pool(sequence_->block_table()) || query_heads_ == 0 ||
        query_heads_ % storage_.layout().num_kv_heads() != 0) {
        throw std::invalid_argument("invalid sequence lifecycle pool or query head count");
    }
}

const SequenceState& SequenceLifecycle::sequence() const {
    if (finished()) throw std::logic_error("request has finished");
    return *sequence_;
}

void SequenceLifecycle::prefill(std::size_t token_count, const void* key_device,
                                const void* value_device, const float* query_device,
                                float* output_device) {
    if (finished() || prefilled_) throw std::logic_error("request cannot prefill again");
    if (token_count == 0 || !key_device || !value_device || !query_device || !output_device) {
        throw std::invalid_argument("prefill requires tokens and device buffers");
    }
    try {
        sequence_->append_tokens(token_count); // 按需申请第一个及后续物理块。
        std::vector<TokenWrite> writes;
        writes.reserve(token_count);
        for (std::size_t token = 0; token < token_count; ++token) {
            writes.push_back({&*sequence_, token});
        }
        const auto slots = build_slot_mapping(writes);
        write_paged_kv(storage_, key_device, value_device, slots);
        paged_attention(storage_, *sequence_, query_device, output_device,
                        token_count, query_heads_);
        prefilled_ = true;
    } catch (...) {
        abort_after_failure();
        throw;
    }
}

TokenLocation SequenceLifecycle::decode(const void* key_device, const void* value_device,
                                        const float* query_device, float* output_device) {
    if (finished() || !prefilled_) throw std::logic_error("request must prefill before decode");
    if (!key_device || !value_device || !query_device || !output_device) {
        throw std::invalid_argument("decode requires device buffers");
    }
    try {
        const auto token_position = sequence_->num_tokens();
        const auto location = append_token_cow(storage_, *sequence_);
        const auto slots = build_slot_mapping({{&*sequence_, token_position}});
        write_paged_kv(storage_, key_device, value_device, slots);
        paged_attention(storage_, *sequence_, query_device, output_device, 1, query_heads_);
        return location;
    } catch (...) {
        abort_after_failure();
        throw;
    }
}

void SequenceLifecycle::finish() noexcept {
    sequence_.reset(); // SequenceState -> BlockTable 析构，逐项归还引用。
    prefilled_ = false;
}

void SequenceLifecycle::abort_after_failure() noexcept {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess &&
        cudaSetDevice(storage_.device()) == cudaSuccess) {
        // 内部读写正常返回时都已同步；异常时避免 GPU 仍访问将被归还的页。
        (void)cudaDeviceSynchronize();
        (void)cudaSetDevice(previous);
    }
    finish();
}

} // namespace kvflux::v2

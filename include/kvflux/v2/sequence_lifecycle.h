#pragma once

#include "kvflux/v2/paged_kv_storage.h"

#include <cstddef>
#include <optional>

namespace kvflux::v2 {

// 单请求、同步的 v2 生命周期。storage 和 pool 必须比本对象活得更久；
// 本类拥有 SequenceState，析构或 finish() 时归还它的全部物理页引用。
class SequenceLifecycle {
public:
    SequenceLifecycle(RequestID request_id, PhysicalBlockPool& pool,
                      PagedKVStorage& storage, std::size_t query_heads);
    ~SequenceLifecycle() noexcept { finish(); }
    SequenceLifecycle(const SequenceLifecycle&) = delete;
    SequenceLifecycle& operator=(const SequenceLifecycle&) = delete;
    SequenceLifecycle(SequenceLifecycle&&) = delete;
    SequenceLifecycle& operator=(SequenceLifecycle&&) = delete;

    // prefill 仅可调用一次。K/V 输入形状为 [token_count][kv_heads][head_size]，
    // Q/O 为 [token_count][query_heads][head_size]；指针均为设备端缓冲。
    // 按 token 顺序分配页、生成 slot mapping、写入全部 KV，再计算因果 attention。
    void prefill(std::size_t token_count, const void* key_device, const void* value_device,
                 const float* query_device, float* output_device);

    // 每次 decode 追加一枚 token。尾块未满时复用；跨块时申请新页；
    // 若尾块被共享，则先复制 K/V 再改本请求的映射。随后写入新 KV 并计算
    // 末尾 Q 的 attention。返回新 token 的物理位置。
    TokenLocation decode(const void* key_device, const void* value_device,
                         const float* query_device, float* output_device);

    // 请求结束时及时释放页引用；重复调用安全。析构时也会执行。
    void finish() noexcept;
    bool finished() const noexcept { return !sequence_.has_value(); }
    const SequenceState& sequence() const;

private:
    // 失败后的 GPU 工作先尽力同步，再释放页；失败的请求不可继续使用。
    void abort_after_failure() noexcept;

    PagedKVStorage& storage_;
    std::optional<SequenceState> sequence_;
    std::size_t query_heads_;
    bool prefilled_ = false;
};

} // namespace kvflux::v2

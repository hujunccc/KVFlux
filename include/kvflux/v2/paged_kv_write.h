#pragma once

#include "kvflux/v2/paged_kv_storage.h"

namespace kvflux::v2 {

// 将设备端 K_new/V_new [batch][num_kv_heads][head_size] 写入 slot_mapping 指定的页。
// 输入 dtype 与 storage.layout().dtype() 相同；slot 数量就是 batch 大小。
// 每次调用上传 slot_mapping、启动 CUDA kernel，并等待完成后返回。
// 输入必须在默认 stream 上已就绪，或由调用方先同步产生输入的其他 stream。
// 空 batch 不访问输入指针。拒绝空指针、越界或重复 slot、错误设备指针。
// 调用方必须保持请求的物理页引用，且不能向仍被其他请求共享的页写入。
void write_paged_kv(PagedKVStorage& storage, const void* key_new_device,
                    const void* value_new_device, const SlotMapping& slot_mapping);

} // namespace kvflux::v2

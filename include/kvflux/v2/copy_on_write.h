#pragma once

#include "kvflux/v2/paged_kv_storage.h"

namespace kvflux::v2 {

// 单 token decode 的设备端写时复制入口。只有共享且未满的尾页会复制；
// 其他情况按普通 append_token 预留位置。返回位置尚未写入新 token 的 K/V。
// 默认 stream 中已有的源页写入须先完成；本函数同步等待 K/V 拷贝完成。
TokenLocation append_token_cow(const PagedKVStorage& storage, SequenceState& sequence);

} // namespace kvflux::v2

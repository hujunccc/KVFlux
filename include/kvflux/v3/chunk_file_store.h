#pragma once

#include "kvflux/v3/cache_directory.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kvflux::v3 {

// 写入 NVMe chunk 文件失败（I/O 错误、配额不足之外的底层错误）。
class ChunkStoreError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// 文件损坏或不属于当前配置：magic/版本/校验和/身份字段任一不匹配。
class ChunkCorruptError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// 目录登记了 NVMe 副本但文件不存在。
class ChunkMissError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// 存储配额（模拟磁盘容量上限）不足以写入新 chunk。
class ChunkQuotaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// chunk 文件头中的元数据（FileMetadata）。身份字段与 CacheKey 一一对应，
// 读取时逐项精确比较：错误模型、布局版本、dtype、层范围或前缀都不能加载。
struct ChunkFileMeta {
    std::string model_id;
    std::uint32_t layout_version = 0;
    DType dtype = DType::Float16;
    std::uint32_t first_layer = 0;
    std::uint32_t last_layer = 0;
    std::uint64_t prefix_hash = 0;
    Tokens prefix;
    std::uint64_t valid_token_count = 0;
    std::uint64_t payload_size = 0;
};

ChunkFileMeta make_file_meta(const CacheKey& key, std::uint64_t valid_tokens,
                             std::uint64_t payload_size);

// ChecksumVerifier：FNV-1a 64 位，覆盖 payload 原始字节；头部另有独立校验。
std::uint64_t chunk_checksum(const void* data, std::size_t bytes) noexcept;

// 基于文件的 chunk 存储。布局：
//   root/chunks/chunk_000001.bin   已发布的完整 chunk
//   root/chunks/chunk_000001.bin.tmp   写入中的临时文件，永不被读取
// 发布用临时文件 + fsync + 原子 rename，读不到写一半的数据。
// 真实 ENOSPC 之外的容量控制由配额模拟；slot 即文件编号。单线程使用。
class ChunkFileStore {
public:
    // root_dir 不存在时创建。quota_bytes 为存储容量上限（模拟磁盘）。
    ChunkFileStore(std::string root_dir, std::uint64_t quota_bytes);

    const std::string& root_dir() const noexcept { return root_dir_; }
    std::string path_for(std::size_t slot) const;
    std::uint64_t stored_bytes() const noexcept { return stored_bytes_; }
    std::uint64_t quota_bytes() const noexcept { return quota_bytes_; }

    // 写入并发布。配额不足抛 ChunkQuotaError；I/O 失败抛 ChunkStoreError。
    // 失败时残留 .tmp 文件不影响 exists/read。
    void write(std::size_t slot, const ChunkFileMeta& meta, const void* payload);

    // 读取并校验。文件不存在返回 false；损坏/版本/身份不匹配抛
    // ChunkCorruptError；payload_size 超过 out_capacity 抛 ChunkCorruptError。
    // 成功时把文件中的完整元数据写入 file_meta（可空）。
    bool read(std::size_t slot, const ChunkFileMeta& expected, void* out,
              std::size_t out_capacity, ChunkFileMeta* file_meta = nullptr) const;

    bool exists(std::size_t slot) const;
    // 删除已发布文件并释放配额；不存在时抛 ChunkMissError。
    void erase(std::size_t slot);

    static constexpr std::uint32_t format_version = 1;

private:
    std::string root_dir_;
    std::uint64_t quota_bytes_;
    std::uint64_t stored_bytes_ = 0;
};

} // namespace kvflux::v3

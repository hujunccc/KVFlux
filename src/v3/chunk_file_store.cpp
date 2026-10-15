#include "kvflux/v3/chunk_file_store.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include <limits>
#include <filesystem>
#include <utility>
#include <vector>

namespace kvflux::v3 {
namespace {

constexpr unsigned char kMagic[4] = {'K', 'V', 'F', 'X'};

class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    ~Fd() { if (fd_ >= 0) ::close(fd_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int get() const { return fd_; }
    int release() { const int f = fd_; fd_ = -1; return f; }
private:
    int fd_ = -1;
};

[[noreturn]] void io_error(const char* operation, const std::string& path) {
    throw ChunkStoreError(std::string(operation) + " failed for " + path + ": " +
                          std::strerror(errno));
}

void write_all(int fd, const void* data, std::size_t bytes, const std::string& path) {
    const auto* p = static_cast<const unsigned char*>(data);
    while (bytes > 0) {
        const auto n = ::write(fd, p, bytes);
        if (n < 0) io_error("write", path);
        p += n;
        bytes -= static_cast<std::size_t>(n);
    }
}

void read_all(int fd, void* data, std::size_t bytes, const std::string& path) {
    auto* p = static_cast<unsigned char*>(data);
    while (bytes > 0) {
        const auto n = ::read(fd, p, bytes);
        if (n < 0) io_error("read", path);
        if (n == 0) throw ChunkCorruptError("truncated chunk file: " + path);
        p += n;
        bytes -= static_cast<std::size_t>(n);
    }
}

void put_u32(std::vector<unsigned char>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<unsigned char>(v >> (8 * i)));
}

void put_u64(std::vector<unsigned char>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<unsigned char>(v >> (8 * i)));
}

std::uint32_t take_u32(const unsigned char*& p) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(*p++) << (8 * i);
    return v;
}

std::uint64_t take_u64(const unsigned char*& p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(*p++) << (8 * i);
    return v;
}

std::vector<unsigned char> serialize_header(const ChunkFileMeta& meta) {
    std::vector<unsigned char> h;
    h.insert(h.end(), kMagic, kMagic + 4);
    put_u32(h, ChunkFileStore::format_version);
    put_u32(h, static_cast<std::uint32_t>(meta.model_id.size()));
    h.insert(h.end(), meta.model_id.begin(), meta.model_id.end());
    put_u32(h, meta.layout_version);
    put_u32(h, static_cast<std::uint32_t>(meta.dtype));
    put_u32(h, meta.first_layer);
    put_u32(h, meta.last_layer);
    put_u64(h, meta.prefix_hash);
    put_u64(h, meta.prefix.size());
    for (const auto token : meta.prefix) put_u32(h, static_cast<std::uint32_t>(token));
    put_u64(h, meta.valid_token_count);
    put_u64(h, meta.payload_size);
    put_u64(h, 0); // payload checksum 占位，写入前回填
    return h;
}

} // namespace

ChunkFileMeta make_file_meta(const CacheKey& key, std::uint64_t valid_tokens,
                             std::uint64_t payload_size) {
    ChunkFileMeta meta;
    meta.model_id = key.model_id;
    meta.layout_version = key.layout_version;
    meta.dtype = key.dtype;
    meta.first_layer = key.first_layer;
    meta.last_layer = key.last_layer;
    meta.prefix_hash = key.prefix_hash;
    meta.prefix = key.prefix;
    meta.valid_token_count = valid_tokens;
    meta.payload_size = payload_size;
    return meta;
}

std::uint64_t chunk_checksum(const void* data, std::size_t bytes) noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

ChunkFileStore::ChunkFileStore(std::string root_dir, std::uint64_t quota_bytes)
    : root_dir_(std::move(root_dir)), quota_bytes_(quota_bytes) {
    std::error_code ec;
    std::filesystem::create_directories(root_dir_ + "/chunks", ec);
    if (ec) throw ChunkStoreError("mkdir failed for " + root_dir_ + ": " + ec.message());
    // 重启后恢复配额占用：只统计已发布的 chunk 文件。
    const auto chunks = root_dir_ + "/chunks";
    DIR* dir = ::opendir(chunks.c_str());
    if (!dir) io_error("opendir", chunks);
    while (auto* entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".bin") != 0) continue;
        struct stat st {};
        if (::stat((chunks + "/" + name).c_str(), &st) == 0) {
            stored_bytes_ += static_cast<std::uint64_t>(st.st_size);
        }
    }
    ::closedir(dir);
}

std::string ChunkFileStore::path_for(std::size_t slot) const {
    char name[64];
    std::snprintf(name, sizeof(name), "/chunks/chunk_%06zu.bin", slot);
    return root_dir_ + name;
}

void ChunkFileStore::write(std::size_t slot, const ChunkFileMeta& meta, const void* payload) {
    auto header = serialize_header(meta);
    const auto payload_sum = chunk_checksum(payload, meta.payload_size);
    // 回填 payload checksum，再追加 header checksum。
    const std::size_t checksum_field = header.size() - 8;
    for (int i = 0; i < 8; ++i) {
        header[checksum_field + i] = static_cast<unsigned char>(payload_sum >> (8 * i));
    }
    put_u64(header, chunk_checksum(header.data(), header.size()));

    const auto final_path = path_for(slot);
    const std::uint64_t file_bytes = header.size() + meta.payload_size;
    std::uint64_t replaced_bytes = 0;
    struct stat old {};
    if (::stat(final_path.c_str(), &old) == 0) {
        replaced_bytes = static_cast<std::uint64_t>(old.st_size);
    }
    if (file_bytes > quota_bytes_ ||
        stored_bytes_ - replaced_bytes + file_bytes > quota_bytes_) {
        throw ChunkQuotaError("chunk store quota exceeded");
    }

    const auto tmp_path = final_path + ".tmp";
    Fd fd(::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644));
    if (fd.get() < 0) io_error("open", tmp_path);
    try {
        write_all(fd.get(), header.data(), header.size(), tmp_path);
        write_all(fd.get(), payload, meta.payload_size, tmp_path);
        if (::fsync(fd.get()) != 0) io_error("fsync", tmp_path);
    } catch (...) {
        const int f = fd.release();
        ::close(f);
        ::unlink(tmp_path.c_str());
        throw;
    }
    if (::close(fd.release()) != 0) io_error("close", tmp_path);
    if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) io_error("rename", final_path);
    // 让 rename 本身落盘，避免重启后出现元数据已见但数据未持久的情况。
    Fd dir_fd(::open((root_dir_ + "/chunks").c_str(), O_RDONLY | O_DIRECTORY));
    if (dir_fd.get() >= 0) ::fsync(dir_fd.get());
    stored_bytes_ = stored_bytes_ - replaced_bytes + file_bytes;
}

bool ChunkFileStore::read(std::size_t slot, const ChunkFileMeta& expected, void* out,
                          std::size_t out_capacity, ChunkFileMeta* file_meta) const {
    const auto path = path_for(slot);
    Fd fd(::open(path.c_str(), O_RDONLY));
    if (fd.get() < 0) {
        if (errno == ENOENT) return false;
        io_error("open", path);
    }

    // 固定前导：magic + version + model_id_len。
    unsigned char lead[12];
    read_all(fd.get(), lead, sizeof(lead), path);
    if (std::memcmp(lead, kMagic, 4) != 0) {
        throw ChunkCorruptError("bad chunk file magic: " + path);
    }
    const unsigned char* p = lead + 4;
    const auto version = take_u32(p);
    if (version != format_version) {
        throw ChunkCorruptError("unsupported chunk file version: " + path);
    }
    const auto model_len = take_u32(p);
    if (model_len > 4096) throw ChunkCorruptError("implausible model id length: " + path);

    ChunkFileMeta meta;
    meta.model_id.resize(model_len);
    read_all(fd.get(), meta.model_id.data(), model_len, path);

    // 头部其余定长字段：layout_version, dtype, layers, prefix_hash, prefix_len。
    unsigned char fixed[32];
    read_all(fd.get(), fixed, sizeof(fixed), path);
    p = fixed;
    meta.layout_version = take_u32(p);
    const auto dtype = take_u32(p);
    if (dtype > 2) throw ChunkCorruptError("unknown dtype in chunk file: " + path);
    meta.dtype = static_cast<DType>(dtype);
    meta.first_layer = take_u32(p);
    meta.last_layer = take_u32(p);
    meta.prefix_hash = take_u64(p);
    const auto prefix_len = take_u64(p);
    if (prefix_len > (1 << 22)) throw ChunkCorruptError("implausible prefix length: " + path);

    meta.prefix.resize(prefix_len);
    std::vector<unsigned char> token_bytes;
    if (prefix_len > 0) {
        token_bytes.resize(prefix_len * 4);
        read_all(fd.get(), token_bytes.data(), token_bytes.size(), path);
        p = token_bytes.data();
        for (auto& token : meta.prefix) {
            token = static_cast<Token>(take_u32(p));
        }
    }

    unsigned char tail[24]; // valid_tokens, payload_size, payload_checksum
    read_all(fd.get(), tail, sizeof(tail), path);
    p = tail;
    meta.valid_token_count = take_u64(p);
    meta.payload_size = take_u64(p);
    const auto payload_sum = take_u64(p);

    unsigned char sum_bytes[8];
    read_all(fd.get(), sum_bytes, sizeof(sum_bytes), path);
    p = sum_bytes;
    const auto header_sum = take_u64(p);

    // 重建完整头部校验 header checksum（header 内的 payload checksum 字段
    // 已回填，因此直接对原文字节求和即可）。
    auto rebuilt = serialize_header(meta);
    const std::size_t checksum_field = rebuilt.size() - 8;
    for (int i = 0; i < 8; ++i) {
        rebuilt[checksum_field + i] = static_cast<unsigned char>(payload_sum >> (8 * i));
    }
    if (chunk_checksum(rebuilt.data(), rebuilt.size()) != header_sum) {
        throw ChunkCorruptError("chunk header checksum mismatch: " + path);
    }

    // 身份逐项精确比较：错误模型/布局/dtype/层范围/前缀不能加载。
    if (meta.model_id != expected.model_id ||
        meta.layout_version != expected.layout_version || meta.dtype != expected.dtype ||
        meta.first_layer != expected.first_layer || meta.last_layer != expected.last_layer ||
        meta.prefix_hash != expected.prefix_hash || meta.prefix != expected.prefix) {
        throw ChunkCorruptError("chunk file identity mismatch: " + path);
    }
    if (meta.payload_size > out_capacity) {
        throw ChunkCorruptError("chunk payload exceeds destination capacity: " + path);
    }

    read_all(fd.get(), out, meta.payload_size, path);
    if (chunk_checksum(out, meta.payload_size) != payload_sum) {
        throw ChunkCorruptError("chunk payload checksum mismatch: " + path);
    }
    if (file_meta) *file_meta = std::move(meta);
    return true;
}

bool ChunkFileStore::exists(std::size_t slot) const {
    struct stat st {};
    return ::stat(path_for(slot).c_str(), &st) == 0;
}

void ChunkFileStore::erase(std::size_t slot) {
    const auto path = path_for(slot);
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        throw ChunkMissError("chunk file does not exist: " + path);
    }
    if (::unlink(path.c_str()) != 0) io_error("unlink", path);
    stored_bytes_ -= static_cast<std::uint64_t>(st.st_size);
}

} // namespace kvflux::v3

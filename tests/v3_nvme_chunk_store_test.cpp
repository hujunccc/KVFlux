#include "kvflux/v3/chunk_file_store.h"
#include "kvflux/v3/host_buffers.h"
#include "kvflux/v3/nvme_transfer.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

using kvflux::Tokens;
using kvflux::v3::CacheDirectory;
using kvflux::v3::CacheKey;
using kvflux::v3::ChunkFileMeta;
using kvflux::v3::ChunkFileStore;
using kvflux::v3::ChunkState;
using kvflux::v3::CpuTier;
using kvflux::v3::GpuTier;
using kvflux::v3::NvmeTier;
using kvflux::v3::StorageTier;

namespace {

constexpr std::size_t BYTES = 256 << 10;

const std::string& test_root() {
    static const std::string root =
        "/tmp/kvflux_v3_nvme_test_" + std::to_string(::getpid());
    return root;
}

struct Fixture {
    kvflux::PhysicalBlockPool gpu_pool{4, 16};
    GpuTier gpu{gpu_pool, BYTES};
    CpuTier cpu{4, BYTES};
    NvmeTier nvme{8, BYTES};
    CacheDirectory dir{gpu, cpu, nvme};
    kvflux::v3::HeapChunkPool cpu_bufs{4, BYTES};
};

CacheKey make_key(std::uint64_t tag, const char* model = "model-x") {
    return kvflux::v3::make_cache_key(model, 1, kvflux::DType::Float16,
                                      0, 32, Tokens{5, 6, static_cast<kvflux::Token>(tag)});
}

std::vector<unsigned char> pattern(std::uint64_t seed, std::size_t bytes) {
    std::vector<unsigned char> out(bytes);
    std::uint64_t x = seed | 1;
    for (auto& b : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        b = static_cast<unsigned char>(x);
    }
    return out;
}

template<class Error, class Fn> void throws(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

void patch_file_byte(const std::string& path, std::size_t offset) {
    const int fd = ::open(path.c_str(), O_RDWR);
    CHECK(fd >= 0);
    unsigned char b = 0;
    CHECK(::pread(fd, &b, 1, offset) == 1);
    b ^= 0xFF;
    CHECK(::pwrite(fd, &b, 1, offset) == 1);
    ::close(fd);
}

} // namespace

// 7.6-1：CPU → NVMe 写入正确，文件可独立读回。
void cpu_to_nvme_write() {
    Fixture f;
    ChunkFileStore store(test_root() + "/write", 64ull << 20);
    kvflux::v3::NvmeTransfer xfer(f.dir, store, f.cpu_bufs);
    const auto key = make_key(1);
    const auto data = pattern(0x11, BYTES);
    f.dir.create(key);
    const auto cpu_loc = f.dir.place(key, StorageTier::CPU);
    std::memcpy(f.cpu_bufs.data(cpu_loc.slot), data.data(), BYTES);

    xfer.offload(key, 16, false);
    const auto& record = f.dir.at(key);
    CHECK(record.nvme->state == ChunkState::READY);
    CHECK(record.cpu->state == ChunkState::READY); // release_cpu=false
    CHECK(store.exists(record.nvme->location.slot));

    std::vector<unsigned char> back(BYTES);
    ChunkFileMeta file_meta;
    const auto expected = kvflux::v3::make_file_meta(key, 0, BYTES);
    CHECK(store.read(record.nvme->location.slot, expected, back.data(), BYTES, &file_meta));
    CHECK(back == data);
    CHECK(file_meta.valid_token_count == 16);
    CHECK(file_meta.payload_size == BYTES);
    CHECK(file_meta.model_id == "model-x");
}

// 7.6-2：NVMe → CPU 读取正确，release 后文件删除。
void nvme_to_cpu_read() {
    Fixture f;
    ChunkFileStore store(test_root() + "/read", 64ull << 20);
    kvflux::v3::NvmeTransfer xfer(f.dir, store, f.cpu_bufs);
    const auto key = make_key(2);
    const auto data = pattern(0x22, BYTES);
    f.dir.create(key);
    const auto cpu_loc = f.dir.place(key, StorageTier::CPU);
    std::memcpy(f.cpu_bufs.data(cpu_loc.slot), data.data(), BYTES);
    xfer.offload(key, 16, true);
    CHECK(!f.dir.at(key).cpu.has_value());
    const auto nvme_slot = f.dir.at(key).nvme->location.slot;

    xfer.onload(key, true);
    const auto& record = f.dir.at(key);
    CHECK(!record.nvme.has_value());
    CHECK(!store.exists(nvme_slot));
    CHECK(record.cpu->state == ChunkState::READY);
    CHECK(std::memcmp(f.cpu_bufs.data(record.cpu->location.slot), data.data(), BYTES) == 0);
    // 写读完成后所有槽位归还。
    xfer.offload(key, 16, true);
    CHECK(f.nvme.allocated_chunks() == 1);
}

// 7.6-4：payload 或 header 损坏都能检测。
void corrupted_chunk_detected() {
    Fixture f;
    ChunkFileStore store(test_root() + "/corrupt", 64ull << 20);
    kvflux::v3::NvmeTransfer xfer(f.dir, store, f.cpu_bufs);
    const auto key = make_key(4);
    const auto data = pattern(0x44, BYTES);
    f.dir.create(key);
    const auto cpu_loc = f.dir.place(key, StorageTier::CPU);
    std::memcpy(f.cpu_bufs.data(cpu_loc.slot), data.data(), BYTES);
    xfer.offload(key, 16, true);
    const auto slot = f.dir.at(key).nvme->location.slot;
    const auto path = store.path_for(slot);
    const auto expected = kvflux::v3::make_file_meta(key, 0, BYTES);
    std::vector<unsigned char> out(BYTES);

    patch_file_byte(path, 200); // header 内部
    throws<kvflux::v3::ChunkCorruptError>(
        [&] { store.read(slot, expected, out.data(), BYTES); });

    // 恢复 header，再破坏 payload 最后一个字节。
    patch_file_byte(path, 200);
    struct stat st {};
    CHECK(::stat(path.c_str(), &st) == 0);
    patch_file_byte(path, static_cast<std::size_t>(st.st_size) - 1);
    throws<kvflux::v3::ChunkCorruptError>(
        [&] { store.read(slot, expected, out.data(), BYTES); });
}

// 7.6-5：文件不存在是 cache miss，不是错误数据。
void missing_file_is_cache_miss() {
    Fixture f;
    ChunkFileStore store(test_root() + "/miss", 64ull << 20);
    kvflux::v3::NvmeTransfer xfer(f.dir, store, f.cpu_bufs);
    const auto key = make_key(5);
    const auto expected = kvflux::v3::make_file_meta(key, 0, BYTES);
    std::vector<unsigned char> out(BYTES);
    CHECK(!store.exists(3));
    CHECK(!store.read(3, expected, out.data(), BYTES)); // 从未写入

    // 目录有 NVMe 副本但文件被外部删除：onload 抛 ChunkMissError 并回滚。
    f.dir.create(key);
    const auto cpu_loc = f.dir.place(key, StorageTier::CPU);
    const auto data = pattern(0x55, BYTES);
    std::memcpy(f.cpu_bufs.data(cpu_loc.slot), data.data(), BYTES);
    xfer.offload(key, 16, true);
    const auto slot = f.dir.at(key).nvme->location.slot;
    CHECK(::unlink(store.path_for(slot).c_str()) == 0);

    throws<kvflux::v3::ChunkMissError>([&] { xfer.onload(key, false); });
    const auto& record = f.dir.at(key);
    CHECK(record.nvme->state == ChunkState::READY); // 源副本恢复
    CHECK(!record.cpu.has_value());                 // 目标副本已回滚
}

// 7.6-6：容量不足正确失败，目录不产生半登记状态。
void full_store_fails_cleanly() {
    Fixture f;
    ChunkFileStore store(test_root() + "/quota", BYTES + (BYTES >> 1)); // 约 1.5 个 chunk
    kvflux::v3::NvmeTransfer xfer(f.dir, store, f.cpu_bufs);
    const auto a = make_key(6), b = make_key(7);
    f.dir.create(a);
    f.dir.create(b);
    const auto loc_a = f.dir.place(a, StorageTier::CPU);
    const auto loc_b = f.dir.place(b, StorageTier::CPU);
    std::memcpy(f.cpu_bufs.data(loc_a.slot), pattern(0x66, BYTES).data(), BYTES);
    std::memcpy(f.cpu_bufs.data(loc_b.slot), pattern(0x77, BYTES).data(), BYTES);

    xfer.offload(a, 16, true);
    throws<kvflux::v3::ChunkQuotaError>([&] { xfer.offload(b, 16, true); });
    const auto& record = f.dir.at(b);
    CHECK(!record.nvme.has_value());
    CHECK(record.cpu->state == ChunkState::READY); // 源副本恢复
    CHECK(f.nvme.allocated_chunks() == 1);

    // 重启后配额占用从磁盘恢复。
    ChunkFileStore reopened(test_root() + "/quota", BYTES + (BYTES >> 1));
    CHECK(reopened.stored_bytes() == store.stored_bytes());
    CHECK(reopened.exists(f.dir.at(a).nvme->location.slot));
}

// 7.6-7：错误文件版本、错误模型配置不能加载。
void wrong_version_or_model_rejected() {
    Fixture f;
    ChunkFileStore store(test_root() + "/version", 64ull << 20);
    kvflux::v3::NvmeTransfer xfer(f.dir, store, f.cpu_bufs);
    const auto key = make_key(8);
    f.dir.create(key);
    const auto cpu_loc = f.dir.place(key, StorageTier::CPU);
    std::memcpy(f.cpu_bufs.data(cpu_loc.slot), pattern(0x88, BYTES).data(), BYTES);
    xfer.offload(key, 16, true);
    const auto slot = f.dir.at(key).nvme->location.slot;
    const auto path = store.path_for(slot);
    std::vector<unsigned char> out(BYTES);

    // 其他模型/配置的 key 读这份文件：身份不匹配。
    const auto wrong_model = kvflux::v3::make_file_meta(make_key(8, "model-y"), 0, BYTES);
    throws<kvflux::v3::ChunkCorruptError>(
        [&] { store.read(slot, wrong_model, out.data(), BYTES); });
    auto wrong_layout = kvflux::v3::make_file_meta(key, 0, BYTES);
    wrong_layout.layout_version = 2;
    throws<kvflux::v3::ChunkCorruptError>(
        [&] { store.read(slot, wrong_layout, out.data(), BYTES); });

    // 篡改格式版本字节。
    patch_file_byte(path, 4);
    const auto expected = kvflux::v3::make_file_meta(key, 0, BYTES);
    throws<kvflux::v3::ChunkCorruptError>(
        [&] { store.read(slot, expected, out.data(), BYTES); });
    patch_file_byte(path, 4); // 恢复
    CHECK(store.read(slot, expected, out.data(), BYTES));
}

// 7.6-8：写入中断（只有 .tmp 或半成品）不会被误认为有效缓存。
void interrupted_write_is_not_valid_cache() {
    const auto root = test_root() + "/interrupted";
    ChunkFileStore store(root, 64ull << 20);
    const auto key = make_key(9);
    const auto expected = kvflux::v3::make_file_meta(key, 0, BYTES);
    std::vector<unsigned char> out(BYTES);

    // 完整内容的 .tmp 文件（rename 前崩溃）不算缓存。
    const auto tmp = store.path_for(7) + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0);
    const auto data = pattern(0x99, BYTES);
    CHECK(::write(fd, data.data(), data.size()) == static_cast<ssize_t>(BYTES));
    ::close(fd);
    CHECK(!store.exists(7));
    CHECK(!store.read(7, expected, out.data(), BYTES));

    // 正常写入后 .tmp 不残留。
    Fixture f2;
    kvflux::v3::NvmeTransfer xfer(f2.dir, store, f2.cpu_bufs);
    const auto key2 = make_key(10);
    f2.dir.create(key2);
    const auto cpu_loc = f2.dir.place(key2, StorageTier::CPU);
    std::memcpy(f2.cpu_bufs.data(cpu_loc.slot), data.data(), BYTES);
    xfer.offload(key2, 16, true);
    const auto slot = f2.dir.at(key2).nvme->location.slot;
    struct stat st {};
    CHECK(::stat((store.path_for(slot) + ".tmp").c_str(), &st) != 0);
    CHECK(store.exists(slot));
}

int main() {
    std::filesystem::remove_all(test_root());
    try {
        cpu_to_nvme_write();
        nvme_to_cpu_read();
        corrupted_chunk_detected();
        missing_file_is_cache_miss();
        full_store_fails_cleanly();
        wrong_version_or_model_rejected();
        interrupted_write_is_not_valid_cache();
        std::cout << "V3.2 NVMe chunk store: write/read/corruption/miss/quota/version/atomicity\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(test_root());
        return 1;
    }
    std::filesystem::remove_all(test_root());
}

// V3.2 NVMe chunk I/O 基准：写（含 fsync + 原子 rename）/ 读延迟与带宽，
// 结果写入 CSV。用法：kvflux_v3_nvme_benchmark [输出.csv] [根目录]

#include "kvflux/v3/chunk_file_store.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "nvme_io_benchmark.csv";
    const std::string root = argc > 2 ? argv[2] : "/tmp/kvflux_v3_nvme_bench";
    try {
        std::filesystem::remove_all(root);
        const std::vector<std::size_t> sizes = {64ull << 10, 1ull << 20, 4ull << 20,
                                                16ull << 20};
        const int samples = 7;
        kvflux::v3::ChunkFileStore store(root, 1ull << 32);
        const auto key = kvflux::v3::make_cache_key("model-x", 1, kvflux::DType::Float16,
                                                    0, 32, kvflux::Tokens{1, 2, 3});
        std::vector<unsigned char> payload;

        std::ofstream csv(out_path);
        if (!csv) throw std::runtime_error("cannot open output: " + out_path);
        csv << "transfer_size_bytes,write_latency_ms,read_latency_ms,"
               "write_bandwidth_mbps,read_bandwidth_mbps\n";
        for (const auto size : sizes) {
            payload.assign(size, 0xAB);
            const auto meta = kvflux::v3::make_file_meta(key, 16, size);
            std::vector<unsigned char> out(size);
            std::vector<double> write_ms, read_ms;
            // 预热一次。
            store.write(0, meta, payload.data());
            store.erase(0);
            for (int i = 0; i < samples; ++i) {
                double t0 = now_ms();
                store.write(0, meta, payload.data());
                double t1 = now_ms();
                const bool found = store.read(0, meta, out.data(), size);
                double t2 = now_ms();
                if (!found) throw std::runtime_error("chunk missing after write");
                store.erase(0);
                write_ms.push_back(t1 - t0);
                read_ms.push_back(t2 - t1);
            }
            const auto w = median(std::move(write_ms));
            const auto r = median(std::move(read_ms));
            const auto mbps = [size](double ms) { return size / (ms * 1e3); };
            csv << size << ',' << w << ',' << r << ',' << mbps(w) << ',' << mbps(r) << '\n';
            std::cout << "size=" << (size >> 10) << " KiB write=" << w << " ms ("
                      << mbps(w) << " MB/s) read=" << r << " ms (" << mbps(r) << " MB/s)\n";
        }
        std::filesystem::remove_all(root);
        std::cout << "wrote " << out_path << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}

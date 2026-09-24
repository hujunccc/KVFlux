#include "kvflux/v2/paged_attention.h"
#include "kvflux/v2/paged_kv_read.h"
#include "kvflux/v2/paged_kv_write.h"
#include "kvflux/v2/reference_attention.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef KVFLUX_V2_CUPTI_PROFILE
#include <cupti.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t block_size = 16, kv_heads = 2, query_heads = 4, head_size = 32;
constexpr std::size_t elements_per_token = kv_heads * head_size;
constexpr int warmup_calls = 5, measured_calls = 30;

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc benchmark buffer");
    }
    ~DeviceBuffer() { (void)cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const noexcept { return static_cast<T*>(pointer_); }
private:
    void* pointer_ = nullptr;
};

// 所有计时调用都已在各自 API 内同步；这里测的是调用者感受到的完整延迟，
// 包含 block table/slot 上传、临时分配及 CUDA kernel，排除 fixture 初始化。
struct Fixture {
    Fixture(std::size_t requests, std::size_t tokens, bool fragmented)
        : tokens(tokens), requests(requests), fragmented(fragmented),
          pool(2 * requests * ((tokens + block_size - 1) / block_size), block_size),
          storage(pool, kv_heads, head_size, kvflux::DType::Float32),
          key_device(requests * tokens * elements_per_token),
          value_device(requests * tokens * elements_per_token),
          gathered_key(tokens * elements_per_token), gathered_value(tokens * elements_per_token),
          query_device(requests * query_heads * head_size),
          output_device(requests * query_heads * head_size) {
        const auto blocks = requests * ((tokens + block_size - 1) / block_size);
        if (fragmented) {
            // 留下 B 个被占用的偶数页，让每个可用物理编号都不相邻。
            holders.reserve(2 * blocks);
            for (std::size_t i = 0; i < 2 * blocks; ++i) holders.push_back(pool.allocate());
            for (std::size_t next = 2 * blocks; next > 0; next -= 2) pool.free(holders[next - 1]);
        }
        sequences.reserve(requests);
        for (std::size_t request = 0; request < requests; ++request) {
            sequences.emplace_back(request + 1, pool);
            sequences.back().append_tokens(tokens);
            sequence_ptrs.push_back(&sequences.back());
        }
        require(sequences.front().num_allocated_blocks() == (tokens + block_size - 1) / block_size,
                "wrong block count");
        if (fragmented) {
            for (const auto& sequence : sequences)
                for (std::size_t logical = 0; logical < sequence.num_allocated_blocks(); ++logical)
                    require(sequence.physical_block_id(logical) % 2 == 1,
                            "fragmented workload did not get isolated physical pages");
        }

        host_key.resize(requests * tokens * elements_per_token);
        host_value.resize(host_key.size());
        std::vector<kvflux::v2::TokenWrite> writes;
        writes.reserve(requests * tokens);
        for (std::size_t request = 0; request < requests; ++request)
            for (std::size_t token = 0; token < tokens; ++token) {
                writes.push_back({&sequences[request], token});
                for (std::size_t element = 0; element < elements_per_token; ++element) {
                    const auto i = (request * tokens + token) * elements_per_token + element;
                    host_key[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.01f;
                    host_value[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.02f;
                }
            }
        check(cudaMemcpy(key_device.data(), host_key.data(), host_key.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "upload benchmark K");
        check(cudaMemcpy(value_device.data(), host_value.data(), host_value.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "upload benchmark V");
        kvflux::v2::write_paged_kv(storage, key_device.data(), value_device.data(),
                                   kvflux::v2::build_slot_mapping(writes));

        // 每个 case 在计时前做完整 K/V 回读，防止 benchmark 只测到了错误地址。
        std::vector<float> actual_key(tokens * elements_per_token), actual_value(actual_key.size());
        for (std::size_t request = 0; request < requests; ++request) {
            kvflux::v2::read_paged_kv(storage, sequences[request],
                                      gathered_key.data(), gathered_value.data());
            check(cudaMemcpy(actual_key.data(), gathered_key.data(), actual_key.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), "verify benchmark K");
            check(cudaMemcpy(actual_value.data(), gathered_value.data(), actual_value.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), "verify benchmark V");
            const auto begin = request * tokens * elements_per_token;
            require(std::equal(actual_key.begin(), actual_key.end(), host_key.begin() + begin),
                    "benchmark K roundtrip mismatch");
            require(std::equal(actual_value.begin(), actual_value.end(), host_value.begin() + begin),
                    "benchmark V roundtrip mismatch");
        }
        host_query.resize(requests * query_heads * head_size);
        for (std::size_t i = 0; i < host_query.size(); ++i)
            host_query[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.01f;
        check(cudaMemcpy(query_device.data(), host_query.data(), host_query.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "upload benchmark Q");
        write_slot = kvflux::v2::build_slot_mapping({{&sequences.front(), tokens - 1}});
    }

    ~Fixture() {
        sequences.clear();
        if (fragmented)
            for (std::size_t i = 0; i < holders.size(); i += 2) pool.free(holders[i]);
    }

    void write_last_token() {
        const auto offset = (tokens - 1) * elements_per_token;
        kvflux::v2::write_paged_kv(storage, key_device.data() + offset,
                                   value_device.data() + offset, write_slot);
    }
    void gather() {
        kvflux::v2::read_paged_kv(storage, sequences.front(),
                                  gathered_key.data(), gathered_value.data());
    }
    void decode() {
        kvflux::v2::paged_attention(storage, sequences.front(), query_device.data(),
                                    output_device.data(), 1, query_heads);
    }
    void batch_decode() {
        kvflux::v2::paged_attention_batch(storage, sequence_ptrs,
                                          query_device.data(), output_device.data(), query_heads);
    }
    void verify_attention_output() {
        std::vector<float> output(requests * query_heads * head_size);
        check(cudaMemcpy(output.data(), output_device.data(), output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "verify benchmark attention output");
        // 使用未经过 paged gather 的原始连续 K/V 构造独立正确性基准。
        for (std::size_t request = 0; request < requests; ++request) {
            const auto kv_begin = request * tokens * elements_per_token;
            const auto q_begin = request * query_heads * head_size;
            const std::vector<float> key(host_key.begin() + kv_begin,
                                         host_key.begin() + kv_begin + tokens * elements_per_token);
            const std::vector<float> value(host_value.begin() + kv_begin,
                                           host_value.begin() + kv_begin + tokens * elements_per_token);
            const std::vector<float> query(host_query.begin() + q_begin,
                                           host_query.begin() + q_begin + query_heads * head_size);
            const auto expected = kvflux::v2::reference_attention(
                query, key, value, {1, tokens, query_heads, kv_heads, head_size});
            for (std::size_t i = 0; i < expected.size(); ++i) {
                const auto actual = output[q_begin + i];
                require(std::isfinite(actual) && std::abs(actual - expected[i]) < 1e-5f,
                        "benchmark paged attention differs from contiguous reference");
            }
        }
    }

    std::size_t tokens, requests;
    bool fragmented;
    kvflux::PhysicalBlockPool pool;
    kvflux::v2::PagedKVStorage storage;
    DeviceBuffer<float> key_device, value_device, gathered_key, gathered_value, query_device, output_device;
    std::vector<kvflux::PhysicalBlockHandle> holders;
    std::vector<kvflux::v2::SequenceState> sequences;
    std::vector<const kvflux::v2::SequenceState*> sequence_ptrs;
    kvflux::v2::SlotMapping write_slot;
    std::vector<float> host_key, host_value, host_query;
};

#ifdef KVFLUX_V2_CUPTI_PROFILE
void cupti_check(CUptiResult result, const char* operation) {
    if (result != CUPTI_SUCCESS) {
        const char* message = nullptr;
        (void)cuptiGetResultString(result, &message);
        throw std::runtime_error(std::string(operation) + ": " + (message ? message : "CUPTI error"));
    }
}

struct ActivityRecord {
    std::string kind, name;
    double duration_us;
    std::size_t bytes;
    int grid_x, block_x;
};
std::mutex activity_mutex;
std::vector<ActivityRecord> activity_records;
std::atomic<int> activity_errors{0};

void CUPTIAPI request_activity_buffer(std::uint8_t** buffer, std::size_t* size,
                                      std::size_t* max_records) {
    *size = 1 << 20;
    *buffer = static_cast<std::uint8_t*>(std::malloc(*size));
    *max_records = 0;
    if (!*buffer) ++activity_errors;
}

void CUPTIAPI complete_activity_buffer(CUcontext, std::uint32_t, std::uint8_t* buffer,
                                       std::size_t, std::size_t valid_size) {
    if (valid_size) {
        CUpti_Activity* record = nullptr;
        CUptiResult status = CUPTI_SUCCESS;
        while ((status = cuptiActivityGetNextRecord(buffer, valid_size, &record)) == CUPTI_SUCCESS) {
            ActivityRecord item{};
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel9*>(record);
                item.kind = "kernel";
                item.name = kernel->name ? kernel->name : "unknown";
                item.duration_us = static_cast<double>(kernel->end - kernel->start) / 1000.0;
                item.grid_x = kernel->gridX;
                item.block_x = kernel->blockX;
            } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                const auto* copy = reinterpret_cast<const CUpti_ActivityMemcpy5*>(record);
                item.kind = "memcpy";
                item.name = "cudaMemcpy";
                item.duration_us = static_cast<double>(copy->end - copy->start) / 1000.0;
                item.bytes = copy->bytes;
            } else {
                continue;
            }
            if (item.duration_us < 0 || !std::isfinite(item.duration_us)) ++activity_errors;
            std::lock_guard<std::mutex> lock(activity_mutex);
            activity_records.push_back(std::move(item));
        }
        if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) ++activity_errors;
    }
    std::free(buffer);
}

std::string csv_name(std::string value) {
    // CUDA 模板 kernel 名称包含逗号；在 CSV 中用引号包裹。
    std::string quoted = "\"";
    for (const char character : value) {
        if (character == '"') quoted += '"';
        quoted += character;
    }
    return quoted + '"';
}

class ActivityProfiler {
public:
    explicit ActivityProfiler(const std::filesystem::path& file) : output_(file) {
        output_.exceptions(std::ios::badbit | std::ios::failbit);
        output_ << "operation,sample,kind,name,duration_us,bytes,grid_x,block_x,host_us\n";
        cupti_check(cuptiActivityRegisterCallbacks(request_activity_buffer, complete_activity_buffer),
                    "cuptiActivityRegisterCallbacks");
        cupti_check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
                    "cuptiActivityEnable kernel");
        cupti_check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),
                    "cuptiActivityEnable memcpy");
    }
    ~ActivityProfiler() {
        (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
        (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    }
    template<class Fn> void capture(const char* operation, int sample, Fn fn) {
        const auto start = Clock::now();
        fn();
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize profile");
        const auto host_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        cupti_check(cuptiActivityFlushAll(0), "cuptiActivityFlushAll");
        std::size_t dropped = 0;
        cupti_check(cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped),
                    "cuptiActivityGetNumDroppedRecords");
        require(dropped == 0 && activity_errors == 0, "CUPTI dropped or invalid activity records");
        std::vector<ActivityRecord> records;
        {
            std::lock_guard<std::mutex> lock(activity_mutex);
            records.swap(activity_records);
        }
        require(!records.empty(), "CUPTI did not capture any GPU activity");
        bool found_kernel = false;
        for (const auto& record : records) {
            found_kernel |= record.kind == "kernel";
            output_ << operation << ',' << sample << ',' << record.kind << ','
                    << csv_name(record.name) << ',' << record.duration_us << ',' << record.bytes
                    << ',' << record.grid_x << ',' << record.block_x << ',' << host_us << '\n';
        }
        require(found_kernel, "CUPTI did not capture a kernel for this operation");
    }
private:
    std::ofstream output_;
};

void run_activity_profile(const std::filesystem::path& dir) {
    Fixture single(1, 256, true);
    Fixture batch(16, 256, true);
    for (int i = 0; i < 5; ++i) {
        single.write_last_token();
        single.gather();
        single.decode();
        batch.batch_decode();
    }
    {
        ActivityProfiler profiler(dir / "cupti_activity.csv");
        for (int sample = 0; sample < 10; ++sample) {
            profiler.capture("write_one_token", sample, [&] { single.write_last_token(); });
            profiler.capture("gather_full_kv", sample, [&] { single.gather(); });
            profiler.capture("decode_attention", sample, [&] { single.decode(); });
            profiler.capture("batch_decode_attention", sample, [&] { batch.batch_decode(); });
        }
    }
    single.verify_attention_output();
    batch.verify_attention_output();
    std::cout << "CUPTI activity captured for V2 write/gather/single/batch kernels\n";
}
#endif

template<class Fn>
void measure(const char* operation, Fixture& fixture, Fn fn,
             std::ofstream& samples, std::ofstream& summary) {
    for (int i = 0; i < warmup_calls; ++i) fn();
    std::vector<double> durations;
    durations.reserve(measured_calls);
    for (int i = 0; i < measured_calls; ++i) {
        const auto start = Clock::now();
        fn();
        const auto us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        durations.push_back(us);
        samples << operation << ',' << fixture.tokens << ',' << fixture.requests << ','
                << (fixture.fragmented ? "fragmented" : "contiguous") << ',' << i << ',' << us << '\n';
    }
    std::sort(durations.begin(), durations.end());
    const auto median = (durations[measured_calls / 2 - 1] + durations[measured_calls / 2]) / 2;
    const auto p95 = durations[static_cast<std::size_t>(std::ceil(measured_calls * 0.95)) - 1];
    summary << operation << ',' << fixture.tokens << ',' << fixture.requests << ','
            << (fixture.fragmented ? "fragmented" : "contiguous") << ','
            << median << ',' << p95 << ',' << durations.front() << ',' << durations.back() << '\n';
    std::cout << operation << " tokens=" << fixture.tokens << " batch=" << fixture.requests
              << (fixture.fragmented ? " fragmented" : " contiguous")
              << " median=" << median << " us p95=" << p95 << " us\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: kvflux_v2_benchmark [output_directory]");
        const auto dir = std::filesystem::path(argc == 2 ? argv[1] : "benchmark/results/v2-local");
        std::filesystem::create_directories(dir);
        check(cudaSetDevice(0), "cudaSetDevice");
#ifdef KVFLUX_V2_CUPTI_PROFILE
        run_activity_profile(dir);
        return 0;
#endif
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        int runtime = 0, driver = 0;
        check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        std::ofstream environment(dir / "environment.txt");
        environment.exceptions(std::ios::badbit | std::ios::failbit);
        const auto now = std::time(nullptr);
        environment << "utc=" << std::put_time(std::gmtime(&now), "%FT%TZ")
                    << "\ngpu=" << prop.name << "\ncompute_capability=" << prop.major << '.' << prop.minor
                    << "\nruntime=" << runtime << "\ndriver=" << driver << "\ncompiler=" << __VERSION__
                    << "\nblock_size=" << block_size << "\nkv_heads=" << kv_heads
                    << "\nquery_heads=" << query_heads << "\nhead_size=" << head_size
                    << "\ndtype=FP32\nwarmup_calls=" << warmup_calls
                    << "\nmeasured_calls=" << measured_calls
                    << "\nmetric=host wall time per synchronous public API call"
                    << "\nsetup_and_correctness_check=excluded from timing\n";
        environment.close();
        std::ofstream samples(dir / "samples.csv"), summary(dir / "summary.csv");
        samples.exceptions(std::ios::badbit | std::ios::failbit);
        summary.exceptions(std::ios::badbit | std::ios::failbit);
        samples << "operation,tokens,batch,layout,sample,latency_us\n";
        summary << "operation,tokens,batch,layout,median_us,p95_us,min_us,max_us\n";

        for (const auto tokens : {std::size_t{64}, std::size_t{256}, std::size_t{1024}})
            for (const bool fragmented : {false, true}) {
                Fixture fixture(1, tokens, fragmented);
                measure("write_one_token", fixture, [&] { fixture.write_last_token(); }, samples, summary);
                measure("gather_full_kv", fixture, [&] { fixture.gather(); }, samples, summary);
                measure("decode_attention", fixture, [&] { fixture.decode(); }, samples, summary);
                fixture.verify_attention_output();
            }
        for (const auto requests : {std::size_t{4}, std::size_t{16}})
            for (const bool fragmented : {false, true}) {
                Fixture fixture(requests, 256, fragmented);
                measure("batch_decode_attention", fixture,
                        [&] { fixture.batch_decode(); }, samples, summary);
                fixture.verify_attention_output();
            }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

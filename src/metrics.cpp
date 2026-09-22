#include "kvflux/metrics.h"
#include <ostream>

namespace kvflux {
std::ostream& operator<<(std::ostream& out, const CacheMetrics& m) {
    out << "gpu_blocks_total=" << m.gpu_blocks_total << '\n'
        << "gpu_blocks_used=" << m.gpu_blocks_used << '\n'
        << "gpu_blocks_free=" << m.gpu_blocks_free << '\n'
        << "cpu_cached_blocks=" << m.cpu_cached_blocks << '\n'
        << "logical_blocks=" << m.logical_blocks << '\n'
        << "transferring_blocks=" << m.transferring_blocks << '\n';
    const auto direction = [&](const char* name, const TransferDirectionMetrics& t) {
        out << name << "_bytes=" << t.bytes << '\n'
            << name << "_batches=" << t.batches << '\n'
            << name << "_device_ms=" << t.device_ms << '\n'
            << name << "_latency_us=" << t.latency_us() << '\n'
            << name << "_bandwidth_gbps=" << t.bandwidth_gbps() << '\n';
    };
    direction("gpu_to_cpu", m.transfers.gpu_to_cpu);
    direction("cpu_to_gpu", m.transfers.cpu_to_gpu);
    return out << "transfer_failed_batches=" << m.transfers.failed_batches << '\n'
        << "offload_count=" << m.offload_count << '\n'
        << "load_count=" << m.load_count << '\n'
        << "prefetch_count=" << m.prefetch_count << '\n'
        << "prefetch_skipped=" << m.prefetch_skipped << '\n'
        << "prefetch_hits=" << m.prefetch_hits << '\n'
        << "prefetch_misses=" << m.prefetch_misses << '\n'
        << "prefetch_late=" << m.prefetch_late << '\n'
        << "prefetch_unused=" << m.prefetch_unused << '\n'
        << "prefetch_outstanding=" << m.prefetch_outstanding << '\n'
        << "demand_count=" << m.demand_count << '\n'
        << "gpu_demand_hits=" << m.gpu_demand_hits << '\n'
        << "request_stall_ms=" << m.request_stall_ms << '\n';
}
} // namespace kvflux

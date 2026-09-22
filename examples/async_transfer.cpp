#include "kvflux/async_transfer.h"
#include <iostream>
#include <vector>

int main() {
    try {
        kvflux::GpuMemoryPool pool(4, 4096, 16);
        kvflux::AsyncTransferRuntime runtime(pool); // pool 必须比 runtime 活得更久。
        kvflux::PinnedBuffer staging(2 * pool.block_bytes()), output(staging.size());
        std::vector<unsigned char> normal(staging.size(), 42), result(normal.size());
        const std::vector<kvflux::BlockHandle> blocks{pool.allocate(), pool.allocate()};
        staging.copy_from(normal.data(), normal.size());
        runtime.write_batch(blocks, staging); // 只提交，不等待 GPU 完成。
        // 此处 CPU 可以做其他工作，也可以向 compute_stream 提交独立 kernel。
        runtime.synchronize(); // 之后 staging 才能被 CPU 重用，块才可发布。
        runtime.read_batch(blocks, output);
        runtime.synchronize();
        output.copy_to(result.data(), result.size());
        if (result != normal) throw std::runtime_error("async roundtrip mismatch");
        for (auto block : blocks) pool.release(block);
        std::cout << "async batch roundtrip=exact, remaining=" << pool.remaining_blocks() << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

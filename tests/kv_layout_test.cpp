#include "kvflux/kv_layout.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#define CHECK(c) do { if (!(c)) throw std::runtime_error("check failed: " #c); } while (false)
template<class Error, class F> void throws(F fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

int main() {
    using namespace kvflux;
    try {
        for (auto dtype : {DType::Float16, DType::BFloat16, DType::Float32}) {
            KVBlockLayout layout({2, 3, 5, 4, dtype});
            const auto bytes = dtype == DType::Float32 ? 4U : 2U;
            CHECK(layout.block_bytes() == 2 * 2 * 3 * 4 * 5 * bytes);
            CHECK(layout.value_offset() == layout.block_bytes() / 2);
            // 按布局顺序枚举每个元素，所有偏移必须紧密、唯一且不越界。
            std::size_t expected = 0;
            for (auto kind : {KVKind::Key, KVKind::Value})
                for (std::size_t l = 0; l < 2; ++l)
                    for (std::size_t h = 0; h < 3; ++h)
                        for (std::size_t t = 0; t < 4; ++t)
                            for (std::size_t d = 0; d < 5; ++d) {
                                CHECK(layout.byte_offset(kind, l, h, t, d) == expected);
                                expected += bytes;
                            }
            CHECK(expected == layout.block_bytes());
            const auto plan = layout.capacity_for(layout.block_bytes() * 7 + 3);
            CHECK(plan.total_blocks == 7 && plan.token_capacity == 28);
            CHECK(plan.unused_bytes == 3);
            CHECK(plan.allocated_bytes == 7 * layout.block_bytes());
            CHECK(layout.capacity_for(0).total_blocks == 0);
            CHECK(layout.capacity_for(layout.block_bytes() - 1).total_blocks == 0);
            throws<std::out_of_range>([&] { layout.byte_offset(KVKind::Key, 2, 0, 0, 0); });
            throws<std::out_of_range>([&] { layout.byte_offset(KVKind::Key, 0, 3, 0, 0); });
            throws<std::out_of_range>([&] { layout.byte_offset(KVKind::Key, 0, 0, 4, 0); });
            throws<std::out_of_range>([&] { layout.byte_offset(KVKind::Key, 0, 0, 0, 5); });
            throws<std::invalid_argument>([&] { layout.byte_offset(static_cast<KVKind>(9), 0, 0, 0, 0); });
        }
        KVBlockLayout llama({32, 8, 128, 16, DType::Float16});
        const auto plan = llama.capacity_for(std::size_t{1} << 30);
        CHECK(plan.block_bytes == 2097152);
        CHECK(plan.total_blocks == 512 && plan.token_capacity == 8192);
        CHECK(plan.unused_bytes == 0);
        throws<std::invalid_argument>([] { KVBlockLayout x({0, 1, 1, 1}); });
        throws<std::invalid_argument>([] { KVBlockLayout x({1, 0, 1, 1}); });
        throws<std::invalid_argument>([] { KVBlockLayout x({1, 1, 0, 1}); });
        throws<std::invalid_argument>([] { KVBlockLayout x({1, 1, 1, 0}); });
        throws<std::invalid_argument>([] { KVBlockLayout x({1, 1, 1, 1, static_cast<DType>(9)}); });
        throws<std::overflow_error>([] {
            KVBlockLayout x({std::numeric_limits<std::size_t>::max(), 1, 1, 1});
        });
        std::cout << "KV layout and capacity checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

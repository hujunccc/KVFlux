#include "kvflux/v2/reference_attention.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Error, class Fn> void throws(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

void close_to(float actual, double expected, double tolerance = 1e-6) {
    CHECK(std::abs(static_cast<double>(actual) - expected) < tolerance);
}

int main() {
    try {
        using kvflux::v2::AttentionShape;
        using kvflux::v2::reference_attention;

        // QK^T 全为零时 softmax 均匀；第一、第二行必须遮住未来的 V。
        const auto causal = reference_attention({0, 0, 0}, {0, 0, 0}, {2, 4, 8},
                                                AttentionShape{3, 3, 1, 1, 1});
        close_to(causal[0], 2.0);
        close_to(causal[1], 3.0);
        close_to(causal[2], 14.0 / 3.0);

        // 只传末尾的 Q：位置应是 1，能看到两个 K/V；head_size=4 时 scale=1/2。
        const auto decode = reference_attention({2, 0, 0, 0},
                                                {0, 0, 0, 0, 1, 0, 0, 0},
                                                {0, 0, 0, 0, 10, 0, 0, 0},
                                                AttentionShape{1, 2, 1, 1, 4});
        close_to(decode[0], 10.0 * std::exp(1.0) / (1.0 + std::exp(1.0)));
        close_to(decode[1], 0.0);

        // Q 是四个 K/V 的末尾两个 token，位置分别为 2 和 3。
        const auto suffix = reference_attention({0, 0}, {0, 0, 0, 0}, {1, 2, 3, 4},
                                                AttentionShape{2, 4, 1, 1, 1});
        close_to(suffix[0], 2.0);
        close_to(suffix[1], 2.5);

        // 两个 Q 头共用一个 KV 头；输出顺序仍按 Q 头排列。
        const auto grouped = reference_attention({0, 0, 0, 0},
                                                 {0, 0, 0, 0}, {1, 10, 3, 30},
                                                 AttentionShape{1, 2, 4, 2, 1});
        CHECK((grouped == std::vector<float>{2, 2, 20, 20}));

        // 大 logits 若直接 exp 会溢出；减去行最大值后应得到稳定结果。
        const auto stable = reference_attention({1e20f}, {1e20f, 1e20f}, {2, 4},
                                                AttentionShape{1, 2, 1, 1, 1});
        close_to(stable[0], 3.0);

        CHECK(reference_attention({}, {}, {}, AttentionShape{0, 0, 1, 1, 1}).empty());
        throws<std::invalid_argument>([] {
            reference_attention({1}, {1}, {1}, AttentionShape{2, 1, 1, 1, 1});
        });
        throws<std::invalid_argument>([] {
            reference_attention({1}, {1}, {1}, AttentionShape{1, 1, 1, 0, 1});
        });
        throws<std::invalid_argument>([] {
            reference_attention({1, 2, 3}, {1}, {1}, AttentionShape{1, 1, 3, 2, 1});
        });
        throws<std::invalid_argument>([] {
            reference_attention({1}, {1}, {}, AttentionShape{1, 1, 1, 1, 1});
        });
        throws<std::invalid_argument>([] {
            reference_attention({std::numeric_limits<float>::quiet_NaN()}, {1}, {1},
                                AttentionShape{1, 1, 1, 1, 1});
        });
        throws<std::overflow_error>([] {
            reference_attention({}, {}, {},
                                AttentionShape{1, 1, std::numeric_limits<std::size_t>::max(), 1, 2});
        });
        std::cout << "reference attention: causal, scale, decode, GQA and stable softmax passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

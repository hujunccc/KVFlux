#include "kvflux/kv_layout.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        // 示例 Llama-like GQA 配置；这里不声称对应某个特定模型版本。
        kvflux::KVBlockLayout layout({32, 8, 128, 16, kvflux::DType::Float16});
        std::size_t gib = 1;
        if (argc > 2) throw std::invalid_argument("usage: kvflux_capacity [positive_integer_GiB]");
        if (argc == 2) {
            const std::string arg = argv[1];
            if (arg.empty() || arg.find_first_not_of("0123456789") != std::string::npos) {
                throw std::invalid_argument("budget must be a positive integer in GiB");
            }
            const auto parsed = std::stoull(arg);
            if (!parsed || parsed > std::numeric_limits<std::size_t>::max() / (1ULL << 30)) {
                throw std::out_of_range("budget out of range");
            }
            gib = static_cast<std::size_t>(parsed);
        }
        const auto plan = layout.capacity_for(gib * (std::size_t{1} << 30));
        std::cout << "layout=[2,32,8,16,128], dtype=FP16\n"
                  << "budget_GiB=" << gib << "\nblock_bytes=" << plan.block_bytes
                  << "\ntotal_blocks=" << plan.total_blocks
                  << "\nallocated_bytes=" << plan.allocated_bytes
                  << "\nunused_bytes=" << plan.unused_bytes
                  << "\ntoken_capacity=" << plan.token_capacity << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

#include "kvflux/kv_layout.h"

#include <limits>
#include <initializer_list>
#include <stdexcept>

namespace kvflux {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("KV layout size overflow");
    }
    return a * b;
}
std::size_t dtype_bytes(DType dtype) {
    switch (dtype) {
    case DType::Float16:
    case DType::BFloat16: return 2;
    case DType::Float32: return 4;
    }
    throw std::invalid_argument("unsupported KV dtype");
}
} // namespace

KVBlockLayout::KVBlockLayout(ModelConfig config)
    : config_(config), element_bytes_(dtype_bytes(config.dtype)), block_bytes_(2) {
    if (!config.num_layers || !config.num_heads || !config.head_dim || !config.block_size) {
        throw std::invalid_argument("all KV dimensions must be positive");
    }
    // 每一步都检查溢出，避免乘法绕回后错误分配一个过小的显存池。
    for (auto size : {config.num_layers, config.num_heads, config.block_size,
                      config.head_dim, element_bytes_}) {
        block_bytes_ = multiply(block_bytes_, size);
    }
}

std::size_t KVBlockLayout::byte_offset(KVKind kind, std::size_t layer, std::size_t head,
                                      std::size_t token, std::size_t dimension) const {
    if (kind != KVKind::Key && kind != KVKind::Value) throw std::invalid_argument("invalid KV kind");
    if (layer >= config_.num_layers || head >= config_.num_heads ||
        token >= config_.block_size || dimension >= config_.head_dim) {
        throw std::out_of_range("KV coordinate out of range");
    }
    // 类似多维数组的行主序寻址；构造函数已保证完整乘积不会溢出。
    auto offset = kind == KVKind::Key ? std::size_t{0} : std::size_t{1};
    offset = offset * config_.num_layers + layer;
    offset = offset * config_.num_heads + head;
    offset = offset * config_.block_size + token;
    offset = offset * config_.head_dim + dimension;
    return offset * element_bytes_;
}

CapacityPlan KVBlockLayout::capacity_for(std::size_t budget_bytes) const {
    const auto blocks = budget_bytes / block_bytes_;
    const auto allocated = blocks * block_bytes_; // 向下取整，因此不会超过预算。
    return {block_bytes_, blocks, allocated, budget_bytes - allocated,
            multiply(blocks, config_.block_size)};
}

} // namespace kvflux

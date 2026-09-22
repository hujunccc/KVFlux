#pragma once

#include <cstddef>

namespace kvflux {

enum class DType { Float16, BFloat16, Float32 };
enum class KVKind { Key, Value };

struct ModelConfig {
    std::size_t num_layers;
    // GQA 模型填写 KV head 数，不是 query head 数。
    std::size_t num_heads;
    std::size_t head_dim;
    std::size_t block_size;
    DType dtype = DType::Float16;
};

struct CapacityPlan {
    std::size_t block_bytes;
    std::size_t total_blocks;
    std::size_t allocated_bytes;
    std::size_t unused_bytes;
    // 表示池中可存储的 token 槽位总数，不是模型的最大上下文长度。
    std::size_t token_capacity;
};

// 统一紧密布局：[K/V][layer][kv_head][token][head_dim]，最后一维连续。
// K 包含所有层，然后才是 V；不添加 padding，不涉及实际显存分配。
class KVBlockLayout {
public:
    explicit KVBlockLayout(ModelConfig config);
    const ModelConfig& config() const noexcept { return config_; }
    std::size_t element_bytes() const noexcept { return element_bytes_; }
    std::size_t block_bytes() const noexcept { return block_bytes_; }
    std::size_t value_offset() const noexcept { return block_bytes_ / 2; }
    std::size_t byte_offset(KVKind kind, std::size_t layer, std::size_t head,
                            std::size_t token, std::size_t dimension) const;
    CapacityPlan capacity_for(std::size_t budget_bytes) const;
private:
    ModelConfig config_;
    std::size_t element_bytes_, block_bytes_;
};

} // namespace kvflux

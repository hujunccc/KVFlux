#include "kvflux/v2/reference_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvflux::v2 {
namespace {

std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("reference attention tensor size overflow");
    }
    return a * b;
}

void check_finite(const std::vector<float>& tensor) {
    for (float x : tensor) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("reference attention inputs must be finite");
        }
    }
}

} // namespace

std::vector<float> reference_attention(const std::vector<float>& query,
                                       const std::vector<float>& key,
                                       const std::vector<float>& value,
                                       const AttentionShape& shape) {
    if (shape.query_heads == 0 || shape.kv_heads == 0 || shape.head_size == 0 ||
        shape.query_heads % shape.kv_heads != 0 || shape.query_tokens > shape.kv_tokens) {
        throw std::invalid_argument("invalid reference attention shape");
    }
    const auto query_elements = checked_multiply(
        checked_multiply(shape.query_tokens, shape.query_heads), shape.head_size);
    const auto kv_elements = checked_multiply(
        checked_multiply(shape.kv_tokens, shape.kv_heads), shape.head_size);
    if (query.size() != query_elements || key.size() != kv_elements ||
        value.size() != kv_elements) {
        throw std::invalid_argument("reference attention tensor size does not match shape");
    }
    check_finite(query);
    check_finite(key);
    check_finite(value);

    std::vector<float> output(query_elements, 0.0f);
    if (shape.query_tokens == 0) return output;

    const double scale = 1.0 / std::sqrt(static_cast<double>(shape.head_size));
    const auto queries_per_kv_head = shape.query_heads / shape.kv_heads;
    std::vector<double> scores(shape.kv_tokens);
    for (std::size_t q_token = 0; q_token < shape.query_tokens; ++q_token) {
        // Q 是 K/V 序列的后缀，故未来 token 的 score 被因果 mask 排除。
        const auto last_visible = shape.kv_tokens - shape.query_tokens + q_token;
        for (std::size_t q_head = 0; q_head < shape.query_heads; ++q_head) {
            const auto kv_head = q_head / queries_per_kv_head;
            const auto q_offset = (q_token * shape.query_heads + q_head) * shape.head_size;
            double maximum = -std::numeric_limits<double>::infinity();

            // QK^T，再缩放；只计算未被 causal mask 遮住的位置。
            for (std::size_t kv_token = 0; kv_token <= last_visible; ++kv_token) {
                const auto k_offset = (kv_token * shape.kv_heads + kv_head) * shape.head_size;
                double dot = 0.0;
                for (std::size_t dim = 0; dim < shape.head_size; ++dim) {
                    dot += static_cast<double>(query[q_offset + dim]) * key[k_offset + dim];
                }
                scores[kv_token] = dot * scale;
                maximum = std::max(maximum, scores[kv_token]);
            }

            // 减最大值防止 exp 溢出；分母只包含可见 token。
            double denominator = 0.0;
            for (std::size_t kv_token = 0; kv_token <= last_visible; ++kv_token) {
                scores[kv_token] = std::exp(scores[kv_token] - maximum);
                denominator += scores[kv_token];
            }

            for (std::size_t dim = 0; dim < shape.head_size; ++dim) {
                double result = 0.0;
                for (std::size_t kv_token = 0; kv_token <= last_visible; ++kv_token) {
                    const auto v_offset = (kv_token * shape.kv_heads + kv_head) * shape.head_size;
                    result += (scores[kv_token] / denominator) * value[v_offset + dim];
                }
                output[q_offset + dim] = static_cast<float>(result);
            }
        }
    }
    return output;
}

} // namespace kvflux::v2

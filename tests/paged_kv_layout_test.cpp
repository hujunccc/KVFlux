#include "kvflux/v2/paged_kv_layout.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("check failed at line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Error, class Fn> void throws(Fn fn) {
    bool caught = false;
    try { fn(); } catch (const Error&) { caught = true; }
    CHECK(caught);
}

void dense_k_then_v_layout(kvflux::DType dtype, std::size_t element_bytes) {
    using kvflux::KVKind;
    kvflux::v2::PagedKVLayout layout(3, 2, 4, 3, dtype);
    CHECK(layout.num_blocks() == 3 && layout.num_kv_heads() == 2);
    CHECK(layout.block_size() == 4 && layout.head_size() == 3);
    CHECK(layout.element_bytes() == element_bytes);
    CHECK(layout.page_bytes() == 2 * 4 * 3 * element_bytes);
    CHECK(layout.cache_bytes() == 3 * layout.page_bytes());
    CHECK(layout.total_bytes() == 2 * layout.cache_bytes());
    CHECK(layout.slot_count() == 12);

    std::size_t expected = 0;
    for (auto kind : {KVKind::Key, KVKind::Value}) {
        for (std::size_t block = 0; block < 3; ++block) {
            CHECK(layout.page_byte_offset(kind, block) == expected);
            for (std::size_t head = 0; head < 2; ++head)
                for (std::size_t token = 0; token < 4; ++token)
                    for (std::size_t dim = 0; dim < 3; ++dim) {
                        CHECK(layout.byte_offset(kind, block, head, token, dim) == expected);
                        CHECK(layout.slot_byte_offset(kind, block * 4 + token, head, dim) == expected);
                        expected += element_bytes;
                    }
        }
    }
    CHECK(expected == layout.total_bytes());

    throws<std::out_of_range>([&] { layout.page_byte_offset(KVKind::Key, 3); });
    throws<std::out_of_range>([&] { layout.byte_offset(KVKind::Key, 0, 2, 0, 0); });
    throws<std::out_of_range>([&] { layout.byte_offset(KVKind::Key, 0, 0, 4, 0); });
    throws<std::out_of_range>([&] { layout.byte_offset(KVKind::Key, 0, 0, 0, 3); });
    throws<std::out_of_range>([&] { layout.slot_byte_offset(KVKind::Key, 12, 0, 0); });
    throws<std::invalid_argument>([&] {
        layout.page_byte_offset(static_cast<KVKind>(9), 0);
    });
}

int main() {
    try {
        for (auto dtype : {kvflux::DType::Float16, kvflux::DType::BFloat16}) {
            dense_k_then_v_layout(dtype, 2);
        }
        dense_k_then_v_layout(kvflux::DType::Float32, 4);
        using kvflux::v2::PagedKVLayout;
        using kvflux::DType;
        throws<std::invalid_argument>([] { PagedKVLayout x(0, 1, 1, 1, DType::Float16); });
        throws<std::invalid_argument>([] { PagedKVLayout x(1, 0, 1, 1, DType::Float16); });
        throws<std::invalid_argument>([] { PagedKVLayout x(1, 1, 0, 1, DType::Float16); });
        throws<std::invalid_argument>([] { PagedKVLayout x(1, 1, 1, 0, DType::Float16); });
        throws<std::invalid_argument>([] { PagedKVLayout x(1, 1, 1, 1, static_cast<DType>(9)); });
        throws<std::overflow_error>([] {
            PagedKVLayout x(std::numeric_limits<std::size_t>::max(), 1, 1, 1, DType::Float16);
        });
        std::cout << "paged KV layout: dense K/V offsets and bounds passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

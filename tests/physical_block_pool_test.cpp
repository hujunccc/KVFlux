#include "kvflux/physical_block_pool.h"

#include <iostream>
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

int main() {
    try {
        using namespace kvflux;
        throws<std::invalid_argument>([] { PhysicalBlockPool pool(0, 16); });
        throws<std::invalid_argument>([] { PhysicalBlockPool pool(1024, 0); });

        PhysicalBlockPool pool(1024, 16);
        CHECK(pool.capacity() == 1024);
        CHECK(pool.block_size() == 16);
        CHECK(pool.available() == 1024);

        // 用户给出的流程：先占三个页，归还中间页，然后复用它的物理编号。
        const auto first = pool.allocate();
        const auto middle = pool.allocate();
        const auto third = pool.allocate();
        CHECK(pool.id(first) == 0);
        CHECK(pool.id(middle) == 1);
        CHECK(pool.id(third) == 2);
        CHECK(pool.available() == 1021);
        pool.free(middle);
        CHECK(pool.available() == 1022);
        const auto reused = pool.allocate();
        CHECK(pool.id(reused) == middle.id);
        CHECK(reused.generation != middle.generation);
        CHECK(pool.available() == 1021);
        throws<std::invalid_argument>([&] { pool.id(middle); });
        throws<std::invalid_argument>([&] { pool.free(middle); });

        // 引用未归零时，物理编号仍在使用，不能进入 free list。
        pool.retain(first);
        CHECK(pool.ref_count(first) == 2);
        pool.free(first);
        CHECK(pool.ref_count(first) == 1);
        CHECK(pool.available() == 1021);

        std::vector<PhysicalBlockHandle> rest;
        rest.reserve(1021);
        for (std::size_t i = 0; i < 1021; ++i) {
            auto handle = pool.allocate();
            CHECK(pool.id(handle) == i + 3);
            rest.push_back(handle);
        }
        CHECK(pool.available() == 0);
        throws<CapacityError>([&] { pool.allocate(); });
        throws<std::invalid_argument>([&] { pool.id({1024, 1}); });

        pool.free(first);
        CHECK(pool.available() == 1);
        const auto first_again = pool.allocate();
        CHECK(pool.id(first_again) == first.id);
        CHECK(first_again.generation != first.generation);
        throws<std::invalid_argument>([&] { pool.retain(first); });

        pool.free(first_again);
        pool.free(reused);
        pool.free(third);
        for (auto handle : rest) pool.free(handle);
        CHECK(pool.available() == pool.capacity());
        std::cout << "physical block pool: reuse, references, capacity and stale handles passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

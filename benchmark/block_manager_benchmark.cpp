#include "kvflux/block_manager.h"
#include <chrono>
#include <iostream>
#include <numeric>

int main() {
    kvflux::BlockManager manager(128, 16);
    kvflux::Tokens tokens(512);
    std::iota(tokens.begin(), tokens.end(), 0);
    auto warmup = manager.acquire(tokens);
    manager.release(warmup);
    constexpr int iterations = 10000;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto table = manager.acquire(tokens);
        manager.release(table);
    }
    const auto elapsed = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "warm-prefix acquire+release: " << elapsed / iterations
              << " us/sequence (512 tokens, block_size=16, iterations="
              << iterations << ")\n";
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    size_t bytes = argc > 1 ? std::strtoull(argv[1], nullptr, 10) * 1024 * 1024 : 512ull * 1024 * 1024;
    int rounds = argc > 2 ? std::atoi(argv[2]) : 3;
    std::vector<uint8_t> source(bytes, 0x5a), destination(bytes);
    volatile uint8_t sink = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (int round = 0; round < rounds; ++round) { std::memcpy(destination.data(), source.data(), bytes); sink ^= destination[(size_t) round % bytes]; }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::printf("{\"bytes\":%zu,\"rounds\":%d,\"elapsed_ms\":%.6f,\"effective_GB_per_s\":%.9f,\"sink\":%u}\n", bytes, rounds, seconds * 1000.0, bytes * rounds / 1e9 / seconds, (unsigned) sink);
    return 0;
}

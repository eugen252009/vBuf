#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <sys/syscall.h>

namespace vbuf_bench {
inline uint64_t now_us() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}
inline FILE * trace_file() {
    static FILE * file = [] {
        const char * path = std::getenv("VBUF_BENCH_TRACE");
        if (!path || !*path) return static_cast<FILE *>(nullptr);
        FILE * result = std::fopen(path, "a");
        if (result) std::setvbuf(result, nullptr, _IOLBF, 0);
        return result;
    }();
    return file;
}
inline void event(const char * name, const char * detail = "") {
    FILE * file = trace_file();
    if (!file) return;
    const auto tid = static_cast<long>(syscall(SYS_gettid));
    std::fprintf(file, "EVENT,%s,%llu,%d,%ld,%s\n", name,
        static_cast<unsigned long long>(now_us()), static_cast<int>(getpid()), tid, detail);
}
inline void repack(const char * tensor, const char * source_type, const char * destination,
                   size_t source_bytes, size_t destination_bytes, uint64_t start_us, uint64_t end_us) {
    FILE * file = trace_file();
    if (!file) return;
    std::fprintf(file, "REPACK,%s,%s,%s,%zu,%zu,%llu,%llu,%llu\n", tensor, source_type,
        destination, source_bytes, destination_bytes,
        static_cast<unsigned long long>(start_us), static_cast<unsigned long long>(end_us),
        static_cast<unsigned long long>(end_us - start_us));
}
}

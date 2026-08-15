// Benchmark-only exact-range storage reader for Step 33.
// It deliberately has no vBuf or llama.cpp dependency.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Stats {
    long long read_bytes = -1, rchar = -1, syscr = -1;
    long voluntary_cs = -1, involuntary_cs = -1;
};

static Stats stats_now() {
    Stats s{};
    std::string key;
    std::ifstream io("/proc/self/io");
    while (io >> key) {
        long long value = -1;
        io >> value;
        if (key == "read_bytes:") s.read_bytes = value;
        else if (key == "rchar:") s.rchar = value;
        else if (key == "syscr:") s.syscr = value;
    }
    std::ifstream status("/proc/self/status");
    while (status >> key) {
        std::string value;
        status >> value;
        if (key == "voluntary_ctxt_switches:") s.voluntary_cs = std::stoll(value);
        else if (key == "nonvoluntary_ctxt_switches:") s.involuntary_cs = std::stoll(value);
    }
    return s;
}

struct Range { uint64_t offset = 0, bytes = 0; };

int main(int argc, char ** argv) {
    if (argc < 7 || argc > 8) return 2;
    const char * path = argv[1];
    const Range range{std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10)};
    const uint64_t chunk = std::strtoull(argv[4], nullptr, 10);
    const unsigned workers = std::max(1u, static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)));
    const std::string destination = argv[6];
    const std::string label = argc == 8 ? argv[7] : "reader";
    if (!range.bytes || !chunk || (destination != "discard" && destination != "ram")) return 2;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return 3;
    struct stat st{};
    if (fstat(fd, &st) != 0 || range.offset > static_cast<uint64_t>(st.st_size) || range.bytes > static_cast<uint64_t>(st.st_size) - range.offset) {
        close(fd); return 4;
    }

    std::vector<uint8_t> ram;
    if (destination == "ram") {
        try { ram.resize(static_cast<size_t>(range.bytes)); }
        catch (...) { close(fd); return 5; }
    }

    std::atomic<uint64_t> next{0};
    std::atomic<uint64_t> successful{0}, requests{0}, short_reads{0}, errors{0};
    std::mutex trace_mutex;
    std::vector<std::pair<uint64_t, uint64_t>> first_ranges, last_ranges;
    const auto before = stats_now();
    const auto begin = Clock::now();

    std::vector<std::thread> threads;
    for (unsigned worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            (void) worker;
            std::vector<uint8_t> buffer(destination == "discard" ? static_cast<size_t>(chunk) : 1);
            while (true) {
                const uint64_t start = next.fetch_add(chunk, std::memory_order_relaxed);
                if (start >= range.bytes) return;
                const uint64_t length = std::min<uint64_t>(chunk, range.bytes - start);
                uint8_t * target = destination == "ram" ? ram.data() + start : buffer.data();
                uint64_t done = 0;
                {
                    std::lock_guard lock(trace_mutex);
                    if (first_ranges.size() < 16) first_ranges.emplace_back(range.offset + start, length);
                    last_ranges.emplace_back(range.offset + start, length);
                    if (last_ranges.size() > 16) last_ranges.erase(last_ranges.begin());
                }
                while (done < length) {
                    const ssize_t n = pread(fd, target + done, static_cast<size_t>(length - done), static_cast<off_t>(range.offset + start + done));
                    requests.fetch_add(1, std::memory_order_relaxed);
                    if (n < 0) { errors.fetch_add(1, std::memory_order_relaxed); return; }
                    if (n == 0) { short_reads.fetch_add(1, std::memory_order_relaxed); return; }
                    if (static_cast<uint64_t>(n) < length - done) short_reads.fetch_add(1, std::memory_order_relaxed);
                    done += static_cast<uint64_t>(n);
                    successful.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto & thread : threads) thread.join();
    const auto after = stats_now();
    close(fd);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();

    std::printf("{\"label\":\"%s\",\"path\":\"%s\",\"range_start\":%llu,\"range_bytes\":%llu,\"chunk_bytes\":%llu,\"workers\":%u,\"destination\":\"%s\",\"elapsed_ms\":%.3f,\"successful_bytes\":%llu,\"requests\":%llu,\"short_reads\":%llu,\"errors\":%llu,\"throughput_GBps\":%.6f,\"before_read_bytes\":%lld,\"after_read_bytes\":%lld,\"physical_read_bytes\":%lld,\"before_syscr\":%lld,\"after_syscr\":%lld,\"syscalls\":%lld,\"voluntary_cs\":%lld,\"involuntary_cs\":%lld,\"first_ranges\":[",
        label.c_str(), path, static_cast<unsigned long long>(range.offset), static_cast<unsigned long long>(range.bytes), static_cast<unsigned long long>(chunk), workers, destination.c_str(), elapsed_ms,
        static_cast<unsigned long long>(successful.load()), static_cast<unsigned long long>(requests.load()), static_cast<unsigned long long>(short_reads.load()), static_cast<unsigned long long>(errors.load()),
        elapsed_ms > 0 ? (successful.load() / 1e9) / (elapsed_ms / 1000.0) : 0.0,
        before.read_bytes, after.read_bytes, after.read_bytes >= 0 && before.read_bytes >= 0 ? after.read_bytes - before.read_bytes : -1,
        before.syscr, after.syscr, after.syscr >= 0 && before.syscr >= 0 ? after.syscr - before.syscr : -1,
        after.voluntary_cs >= 0 && before.voluntary_cs >= 0 ? after.voluntary_cs - before.voluntary_cs : -1,
        after.involuntary_cs >= 0 && before.involuntary_cs >= 0 ? after.involuntary_cs - before.involuntary_cs : -1);
    for (size_t i = 0; i < first_ranges.size(); ++i) std::printf("%s{\"offset\":%llu,\"bytes\":%llu}", i ? "," : "", static_cast<unsigned long long>(first_ranges[i].first), static_cast<unsigned long long>(first_ranges[i].second));
    std::printf("],\"last_ranges\":[");
    for (size_t i = 0; i < last_ranges.size(); ++i) std::printf("%s{\"offset\":%llu,\"bytes\":%llu}", i ? "," : "", static_cast<unsigned long long>(last_ranges[i].first), static_cast<unsigned long long>(last_ranges[i].second));
    std::printf("]}\n");
    return errors.load() || short_reads.load() || successful.load() != range.bytes ? 8 : 0;
}

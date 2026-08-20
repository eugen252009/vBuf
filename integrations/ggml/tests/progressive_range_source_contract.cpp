#include "vbuf_range_source.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

class FixtureSource final : public vbuf_ggml::RangeSource {
public:
    explicit FixtureSource(std::vector<uint8_t> bytes, bool truncated = false,
        bool failed = false)
        : bytes_(std::move(bytes)), truncated_(truncated), failed_(failed) {}

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        vbuf_ggml::RangeReadResult * result) override {
        ++calls_;
        {
            std::lock_guard<std::mutex> lock(ranges_mutex_);
            ranges_.emplace_back(offset, length);
        }
        if (result != nullptr) {
            *result = {};
            result->requested_offset = offset;
            result->requested_length = length;
            result->source_id = "fixture-remote";
        }
        if (failed_ || offset + length > bytes_.size()) return false;
        if (delay_) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (truncated_) {
            if (result != nullptr) result->returned_bytes = length - 1;
            return false;
        }
        std::copy(bytes_.begin() + static_cast<ptrdiff_t>(offset),
            bytes_.begin() + static_cast<ptrdiff_t>(offset + length), destination);
        if (result != nullptr) {
            result->returned_bytes = length;
            result->status_code = 206;
        }
        return true;
    }

    uint64_t calls() const { return calls_; }
    void set_delay(bool delay) { delay_ = delay; }
    std::vector<std::pair<uint64_t, uint64_t>> ranges() const {
        std::lock_guard<std::mutex> lock(ranges_mutex_);
        return ranges_;
    }

private:
    std::vector<uint8_t> bytes_;
    bool truncated_;
    bool failed_;
    bool delay_ = false;
    std::atomic<uint64_t> calls_ = 0;
    mutable std::mutex ranges_mutex_;
    std::vector<std::pair<uint64_t, uint64_t>> ranges_;
};

vbuf_ggml::ProgressiveSourceIdentity identity(uint8_t seed = 1, uint64_t size = 10000) {
    vbuf_ggml::ProgressiveSourceIdentity result;
    result.declared_size = size;
    result.hash_algorithm = 1;
    for (size_t index = 0; index < result.full_source_hash.size(); ++index)
        result.full_source_hash[index] = static_cast<uint8_t>(seed + index);
    return result;
}

std::filesystem::path test_path(const char * name) {
    const auto path = std::filesystem::temp_directory_path() /
        (std::string("vbuf-progressive-") + name + "-" + std::to_string(::getpid()));
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".coverage");
    return path;
}

std::vector<uint8_t> fixture_bytes() {
    std::vector<uint8_t> bytes(10000);
    for (size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<uint8_t>((index * 17 + 3) & 0xff);
    return bytes;
}

void assert_slice(const std::vector<uint8_t> & actual, const std::vector<uint8_t> & expected,
    size_t offset) {
    assert(actual.size() <= expected.size() - offset);
    assert(std::equal(actual.begin(), actual.end(), expected.begin() + static_cast<ptrdiff_t>(offset)));
}

void cold_warm_and_boundary_requests() {
    const auto bytes = fixture_bytes();
    const auto path = test_path("cold-warm");
    auto remote = std::make_shared<FixtureSource>(bytes);
    vbuf_ggml::ProgressiveRangeSource source(remote, path.string(), identity());

    std::vector<uint8_t> result(5000);
    vbuf_ggml::RangeReadResult read_result;
    assert(source.read_range(5000, result.size(), result.data(), &read_result));
    assert_slice(result, bytes, 5000);
    assert(remote->calls() == 1);
    assert((remote->ranges()[0] == std::make_pair<uint64_t, uint64_t>(4096, 5904)));
    assert(source.chunk_is_covered(1));
    assert(source.chunk_is_covered(2));
    assert(source.progressive_metrics().chunks_published == 2);
    assert(source.progressive_metrics().upstream_remote_requests == 1);
    assert(source.progressive_metrics().acquisition_windows == 1);
    assert(std::filesystem::file_size(path) == 10000);
    assert(std::filesystem::file_size(path.string() + ".coverage") == 81);

    assert(source.read_range(5000, result.size(), result.data(), &read_result));
    assert_slice(result, bytes, 5000);
    assert(remote->calls() == 1);
    assert(source.progressive_metrics().local_chunk_hits == 2);

    std::vector<uint8_t> first_chunk(4096);
    assert(source.read_range(0, first_chunk.size(), first_chunk.data(), &read_result));
    assert_slice(first_chunk, bytes, 0);
    assert(remote->calls() == 2);
    assert((remote->ranges()[1] == std::make_pair<uint64_t, uint64_t>(0, 4096)));
    assert(!source.read_range(0, 0, first_chunk.data(), &read_result));
    assert(!source.read_range(9999, 2, first_chunk.data(), &read_result));

    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".coverage");
}

void bounded_windows_reduce_upstream_requests() {
    constexpr uint64_t chunk_count = 700;
    const auto bytes = std::vector<uint8_t>(chunk_count * 4096, 0x5a);
    const auto path = test_path("bounded-windows");
    auto remote = std::make_shared<FixtureSource>(bytes);
    vbuf_ggml::ProgressiveRangeSource source(remote, path.string(),
        identity(1, bytes.size()));
    std::vector<uint8_t> result(bytes.size());
    vbuf_ggml::RangeReadResult read_result;
    assert(source.read_range(0, result.size(), result.data(), &read_result));
    assert(result == bytes);
    assert(remote->calls() == 3);
    for (const auto & range : remote->ranges()) assert(range.second <= 1u << 20);
    assert(source.progressive_metrics().chunks_published == chunk_count);
    assert(source.progressive_metrics().upstream_remote_requests == 3);
    assert(source.progressive_metrics().acquisition_windows == 3);
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".coverage");
}

void partial_local_and_disjoint_missing_runs() {
    constexpr uint64_t chunk_count = 12;
    const auto bytes = std::vector<uint8_t>(chunk_count * 4096, 0x31);
    const auto path = test_path("disjoint-runs");
    auto remote = std::make_shared<FixtureSource>(bytes);
    vbuf_ggml::ProgressiveRangeSource source(remote, path.string(),
        identity(1, bytes.size()));
    std::vector<uint8_t> result(4096);
    std::vector<uint8_t> cross_chunk_result(5 * 4096);
    vbuf_ggml::RangeReadResult read_result;
    assert(source.read_range(2 * 4096, result.size(), result.data(), &read_result));
    assert(source.read_range(4096, cross_chunk_result.size(), cross_chunk_result.data(), &read_result));
    assert(remote->calls() == 3);
    const auto ranges = remote->ranges();
    assert((ranges[1] == std::make_pair<uint64_t, uint64_t>(4096, 4096)));
    assert((ranges[2] == std::make_pair<uint64_t, uint64_t>(3 * 4096, 3 * 4096)));
    assert(source.progressive_metrics().local_chunk_hits == 1);
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".coverage");
}

void failed_remote_and_truncated_remote_do_not_publish() {
    const auto bytes = fixture_bytes();
    for (const bool truncated : { false, true }) {
        const auto path = test_path(truncated ? "truncated" : "failed");
        auto remote = std::make_shared<FixtureSource>(bytes, truncated, !truncated);
        vbuf_ggml::ProgressiveRangeSource source(remote, path.string(), identity());
        std::vector<uint8_t> result(64);
        vbuf_ggml::RangeReadResult read_result;
        assert(!source.read_range(100, result.size(), result.data(), &read_result));
        assert(!source.chunk_is_covered(0));
        assert(source.progressive_metrics().chunks_published == 0);
        assert(source.progressive_metrics().upstream_remote_requests == 0);
        std::filesystem::remove(path);
        std::filesystem::remove(path.string() + ".coverage");
    }
}

void failed_local_write_does_not_publish() {
    const auto bytes = std::vector<uint8_t>(3 * 4096, 0x42);
    const auto path = test_path("failed-write");
    auto remote = std::make_shared<FixtureSource>(bytes);
    vbuf_ggml::ProgressiveRangeSource source(remote, path.string(),
        identity(1, bytes.size()));

    struct rlimit previous_limit{};
    assert(getrlimit(RLIMIT_FSIZE, &previous_limit) == 0);
    if (previous_limit.rlim_max != RLIM_INFINITY && previous_limit.rlim_max < 4096) {
        std::filesystem::remove(path);
        std::filesystem::remove(path.string() + ".coverage");
        return;
    }
    const auto previous_signal = std::signal(SIGXFSZ, SIG_IGN);
    assert(previous_signal != SIG_ERR);
    struct rlimit limited = previous_limit;
    limited.rlim_cur = 4096;
    assert(setrlimit(RLIMIT_FSIZE, &limited) == 0);

    std::vector<uint8_t> result(64);
    vbuf_ggml::RangeReadResult read_result;
    assert(!source.read_range(2 * 4096, result.size(), result.data(), &read_result));
    assert(!source.chunk_is_covered(2));
    assert(source.progressive_metrics().chunks_published == 0);
    assert(source.progressive_metrics().upstream_remote_requests == 0);
    assert(source.progressive_metrics().acquisition_windows == 0);

    assert(setrlimit(RLIMIT_FSIZE, &previous_limit) == 0);
    assert(std::signal(SIGXFSZ, previous_signal) != SIG_ERR);
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".coverage");
}

void identity_mismatch_and_reopen_fail_closed() {
    const auto bytes = fixture_bytes();
    const auto path = test_path("identity");
    auto remote = std::make_shared<FixtureSource>(bytes);
    {
        vbuf_ggml::ProgressiveRangeSource source(remote, path.string(), identity());
        std::vector<uint8_t> result(128);
        vbuf_ggml::RangeReadResult read_result;
        assert(source.read_range(4096, result.size(), result.data(), &read_result));
    }
    bool rejected = false;
    try {
        vbuf_ggml::ProgressiveRangeSource mismatch(remote, path.string(), identity(2));
    } catch (...) {
        rejected = true;
    }
    assert(rejected);
    auto size_mismatch = identity();
    size_mismatch.declared_size = 9999;
    rejected = false;
    try {
        vbuf_ggml::ProgressiveRangeSource mismatch(remote, path.string(), size_mismatch);
    } catch (...) {
        rejected = true;
    }
    assert(rejected);

    auto reopened_remote = std::make_shared<FixtureSource>(bytes);
    vbuf_ggml::ProgressiveRangeSource reopened(reopened_remote, path.string(), identity());
    std::vector<uint8_t> result(128);
    vbuf_ggml::RangeReadResult read_result;
    assert(reopened.read_range(4096, result.size(), result.data(), &read_result));
    assert_slice(result, bytes, 4096);
    assert(reopened_remote->calls() == 0);
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".coverage");
}

void concurrent_same_chunk_is_serialized() {
    const auto bytes = fixture_bytes();
    const auto path = test_path("concurrent");
    auto remote = std::make_shared<FixtureSource>(bytes);
    remote->set_delay(true);
    vbuf_ggml::ProgressiveRangeSource source(remote, path.string(), identity());
    std::vector<uint8_t> first(100), second(100);
    vbuf_ggml::RangeReadResult first_result, second_result;
    std::thread a([&] { assert(source.read_range(5000, first.size(), first.data(), &first_result)); });
    std::thread b([&] { assert(source.read_range(5100, second.size(), second.data(), &second_result)); });
    a.join();
    b.join();
    assert_slice(first, bytes, 5000);
    assert_slice(second, bytes, 5100);
    assert(remote->calls() == 1);
    assert(source.progressive_metrics().chunks_published == 1);
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".coverage");
}

void invalid_persistent_store_fails_closed() {
    const auto bytes = fixture_bytes();
    const auto path = test_path("invalid-store");
    std::filesystem::create_directory(path);
    auto remote = std::make_shared<FixtureSource>(bytes);
    bool rejected = false;
    try {
        vbuf_ggml::ProgressiveRangeSource source(remote, path.string(), identity());
    } catch (...) {
        rejected = true;
    }
    assert(rejected);
    std::filesystem::remove(path);
}

} // namespace

int main() {
    cold_warm_and_boundary_requests();
    bounded_windows_reduce_upstream_requests();
    partial_local_and_disjoint_missing_runs();
    failed_remote_and_truncated_remote_do_not_publish();
    failed_local_write_does_not_publish();
    identity_mismatch_and_reopen_fail_closed();
    concurrent_same_chunk_is_serialized();
    invalid_persistent_store_fails_closed();
    return 0;
}

#include "vbuf_range_source.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
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

private:
    std::vector<uint8_t> bytes_;
    bool truncated_;
    bool failed_;
    bool delay_ = false;
    std::atomic<uint64_t> calls_ = 0;
};

vbuf_ggml::ProgressiveSourceIdentity identity(uint8_t seed = 1) {
    vbuf_ggml::ProgressiveSourceIdentity result;
    result.declared_size = 10000;
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
    assert(remote->calls() == 2);
    assert(source.chunk_is_covered(1));
    assert(source.chunk_is_covered(2));
    assert(source.progressive_metrics().chunks_published == 2);
    assert(std::filesystem::file_size(path) == 10000);
    assert(std::filesystem::file_size(path.string() + ".coverage") == 81);

    assert(source.read_range(5000, result.size(), result.data(), &read_result));
    assert_slice(result, bytes, 5000);
    assert(remote->calls() == 2);
    assert(source.progressive_metrics().local_chunk_hits == 2);

    std::vector<uint8_t> first_chunk(4096);
    assert(source.read_range(0, first_chunk.size(), first_chunk.data(), &read_result));
    assert_slice(first_chunk, bytes, 0);
    assert(remote->calls() == 3);
    assert(!source.read_range(0, 0, first_chunk.data(), &read_result));
    assert(!source.read_range(9999, 2, first_chunk.data(), &read_result));

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
        std::filesystem::remove(path);
        std::filesystem::remove(path.string() + ".coverage");
    }
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
    failed_remote_and_truncated_remote_do_not_publish();
    identity_mismatch_and_reopen_fail_closed();
    concurrent_same_chunk_is_serialized();
    invalid_persistent_store_fails_closed();
    return 0;
}

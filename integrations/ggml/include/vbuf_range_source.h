#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace vbuf_ggml {

struct RangeReadResult {
    uint64_t requested_offset = 0;
    uint64_t requested_length = 0;
    uint64_t returned_bytes = 0;
    uint64_t first_byte_timestamp_ns = 0;
    int status_code = 0;
    std::string source_id;
    std::string content_range;
    std::string error;
    std::string local_endpoint;
    std::string remote_endpoint;
};

struct RangeSourceMetrics {
    uint64_t requests = 0;
    uint64_t bytes = 0;
    uint64_t unique_bytes = 0;
    uint64_t connections = 0;
    uint64_t min_request_ns = 0;
    uint64_t median_request_ns = 0;
    uint64_t max_request_ns = 0;
};

class RangeSource {
public:
    virtual ~RangeSource() = default;

    virtual bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) = 0;
};

class LocalVbufRangeSource final : public RangeSource {
public:
    LocalVbufRangeSource(const uint8_t * mapped_base, uint64_t artifact_bytes,
        std::string source_id = "local-vbuf");

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override;

private:
    const uint8_t * mapped_base_;
    uint64_t artifact_bytes_;
    std::string source_id_;
};

class HttpRangeSource final : public RangeSource {
public:
    explicit HttpRangeSource(std::string endpoint, std::string local_source_ip = {});
    ~HttpRangeSource() override;

    const std::string & endpoint() const { return endpoint_; }
    const std::string & local_source_ip() const { return local_source_ip_; }

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override;
    RangeSourceMetrics metrics() const;

private:
    std::string endpoint_;
    std::string local_source_ip_;
    int socket_fd_ = -1;
    mutable std::mutex socket_mutex_;
    std::vector<uint64_t> request_durations_ns_;
    std::unordered_set<std::string> unique_ranges_;
    std::unordered_set<std::string> connection_endpoints_;
    uint64_t transferred_bytes_ = 0;
};

struct ProgressiveSourceIdentity {
    uint64_t declared_size = 0;
    uint16_t hash_algorithm = 0;
    std::array<uint8_t, 32> full_source_hash{};
};

struct ProgressiveRangeMetrics {
    uint64_t local_source_bytes = 0;
    uint64_t remote_source_bytes = 0;
    uint64_t local_chunk_hits = 0;
    uint64_t remote_chunk_misses = 0;
    uint64_t chunks_published = 0;
    uint64_t covered_chunks = 0;
    uint64_t coverage_bytes = 0;
};

// Source-layer local-hit/remote-miss composition for one immutable payload.
// The payload file is canonical-offset sparse storage; the sidecar bitmap is
// the only coverage authority.
class ProgressiveRangeSource final : public RangeSource {
public:
    static constexpr uint64_t CHUNK_SIZE = 4096;

    ProgressiveRangeSource(std::shared_ptr<RangeSource> remote,
        std::string mirror_path, ProgressiveSourceIdentity identity);
    ~ProgressiveRangeSource() override;

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override;

    RangeSourceMetrics metrics() const;
    ProgressiveRangeMetrics progressive_metrics() const;
    const ProgressiveSourceIdentity & identity() const { return identity_; }
    const std::string & mirror_path() const { return mirror_path_; }
    const std::string & coverage_path() const { return coverage_path_; }
    bool chunk_is_covered(uint64_t chunk_index) const;

private:
    bool open_store();
    bool ensure_chunk(uint64_t chunk_index);
    bool publish_chunk(uint64_t chunk_index);
    bool read_local(uint64_t offset, uint64_t length, uint8_t * destination);
    bool write_at(uint64_t offset, const uint8_t * source, size_t length);
    bool read_at(uint64_t offset, uint8_t * destination, size_t length) const;
    bool valid_range(uint64_t offset, uint64_t length, uint64_t * end) const;
    uint64_t chunk_count() const;
    uint64_t chunk_length(uint64_t chunk_index) const;
    bool bit_is_set(uint64_t chunk_index) const;
    void set_bit(uint64_t chunk_index);
    bool read_sidecar_header();
    bool initialize_sidecar();
    bool write_sidecar_header() const;

    std::shared_ptr<RangeSource> remote_;
    std::string mirror_path_;
    std::string coverage_path_;
    ProgressiveSourceIdentity identity_;
    int mirror_fd_ = -1;
    int coverage_fd_ = -1;
    std::vector<uint8_t> coverage_;
    mutable std::mutex mutex_;
    ProgressiveRangeMetrics metrics_{};
};

} // namespace vbuf_ggml

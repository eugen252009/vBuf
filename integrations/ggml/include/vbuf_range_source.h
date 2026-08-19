#pragma once

#include <cstddef>
#include <cstdint>
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

} // namespace vbuf_ggml

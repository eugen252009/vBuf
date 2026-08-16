#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

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
    explicit HttpRangeSource(std::string endpoint);

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override;

private:
    std::string endpoint_;
};

} // namespace vbuf_ggml

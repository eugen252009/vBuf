#include "vbuf_range_source.h"
#include "vbuf_d0_1_diagnostics.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

namespace vbuf_ggml {
namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

RangeSourceMetrics summarize_metrics(const std::vector<uint64_t> & durations,
    uint64_t bytes, const std::unordered_set<std::string> & unique_ranges,
    const std::unordered_set<std::string> & connections) {
    RangeSourceMetrics metrics{};
    metrics.requests = durations.size();
    metrics.bytes = bytes;
    metrics.connections = connections.size();
    for (const std::string & range : unique_ranges) {
        const size_t separator = range.find(':');
        metrics.unique_bytes += std::stoull(range.substr(separator + 1));
    }
    if (!durations.empty()) {
        std::vector<uint64_t> sorted = durations;
        std::sort(sorted.begin(), sorted.end());
        metrics.min_request_ns = sorted.front();
        metrics.median_request_ns = sorted[sorted.size() / 2];
        metrics.max_request_ns = sorted.back();
    }
    return metrics;
}

bool checked_end(uint64_t offset, uint64_t length, uint64_t * end) {
    if (length == 0 || offset > std::numeric_limits<uint64_t>::max() - length) return false;
    *end = offset + length;
    return true;
}

constexpr std::array<uint8_t, 8> PROGRESSIVE_COVERAGE_MAGIC = {
    'V', 'B', 'U', 'F', 'P', 'C', 'O', 'V' };
constexpr uint32_t PROGRESSIVE_COVERAGE_VERSION = 1;
constexpr uint64_t PROGRESSIVE_COVERAGE_HEADER_BYTES = 80;

void put_u16(uint8_t * destination, uint16_t value) {
    destination[0] = static_cast<uint8_t>(value);
    destination[1] = static_cast<uint8_t>(value >> 8);
}

void put_u32(uint8_t * destination, uint32_t value) {
    for (size_t index = 0; index < 4; ++index)
        destination[index] = static_cast<uint8_t>(value >> (index * 8));
}

void put_u64(uint8_t * destination, uint64_t value) {
    for (size_t index = 0; index < 8; ++index)
        destination[index] = static_cast<uint8_t>(value >> (index * 8));
}

uint16_t get_u16(const uint8_t * source) {
    return static_cast<uint16_t>(source[0]) |
        static_cast<uint16_t>(source[1]) << 8;
}

uint32_t get_u32(const uint8_t * source) {
    uint32_t value = 0;
    for (size_t index = 0; index < 4; ++index)
        value |= static_cast<uint32_t>(source[index]) << (index * 8);
    return value;
}

uint64_t get_u64(const uint8_t * source) {
    uint64_t value = 0;
    for (size_t index = 0; index < 8; ++index)
        value |= static_cast<uint64_t>(source[index]) << (index * 8);
    return value;
}

bool positioned_write(int fd, const uint8_t * source, size_t length, uint64_t offset) {
    size_t written = 0;
    while (written < length) {
        const ssize_t count = pwrite(fd, source + written, length - written,
            static_cast<off_t>(offset + written));
        if (count <= 0) return false;
        written += static_cast<size_t>(count);
    }
    return true;
}

bool positioned_read(int fd, uint8_t * destination, size_t length, uint64_t offset) {
    size_t read_bytes = 0;
    while (read_bytes < length) {
        const ssize_t count = pread(fd, destination + read_bytes, length - read_bytes,
            static_cast<off_t>(offset + read_bytes));
        if (count <= 0) return false;
        read_bytes += static_cast<size_t>(count);
    }
    return true;
}

} // namespace

LocalVbufRangeSource::LocalVbufRangeSource(const uint8_t * mapped_base,
    uint64_t artifact_bytes, std::string source_id)
    : mapped_base_(mapped_base), artifact_bytes_(artifact_bytes), source_id_(std::move(source_id)) {}

bool LocalVbufRangeSource::read_range(uint64_t offset, uint64_t length,
    uint8_t * destination, RangeReadResult * result) {
    *result = {};
    result->requested_offset = offset;
    result->requested_length = length;
    result->source_id = source_id_;
    uint64_t end = 0;
    if (destination == nullptr || mapped_base_ == nullptr || !checked_end(offset, length, &end) ||
        end > artifact_bytes_) {
        result->error = "local range is outside the validated artifact";
        return false;
    }
    result->first_byte_timestamp_ns = now_ns();
    std::memcpy(destination, mapped_base_ + offset, static_cast<size_t>(length));
    result->returned_bytes = length;
    result->status_code = 200;
    return true;
}

HttpRangeSource::HttpRangeSource(std::string endpoint, std::string local_source_ip)
    : endpoint_(std::move(endpoint)), local_source_ip_(std::move(local_source_ip)) {}

HttpRangeSource::~HttpRangeSource() {
    if (socket_fd_ >= 0) close(socket_fd_);
}

bool HttpRangeSource::read_range(uint64_t offset, uint64_t length,
    uint8_t * destination, RangeReadResult * result) {
    const uint64_t request_start_ns = now_ns();
    std::unique_lock<std::mutex> socket_lock(socket_mutex_);
    const uint64_t mutex_acquired_ns = now_ns();
    *result = {};
    result->requested_offset = offset;
    result->requested_length = length;
    result->source_id = endpoint_;
    uint64_t end = 0;
    if (destination == nullptr || !checked_end(offset, length, &end)) {
        result->error = "invalid HTTP range";
        return false;
    }
    if (endpoint_.compare(0, 7, "http://") != 0) {
        result->error = "only http:// endpoints are supported by this POC";
        return false;
    }
    const std::string authority_and_path = endpoint_.substr(7);
    const size_t slash = authority_and_path.find('/');
    const std::string authority = authority_and_path.substr(0, slash);
    const std::string path = slash == std::string::npos
        ? "/" : authority_and_path.substr(slash);
    const size_t colon = authority.rfind(':');
    const std::string host = authority.substr(0, colon);
    const std::string port = colon == std::string::npos ? "80" : authority.substr(colon + 1);
    sockaddr_in local_address{};
    if (!local_source_ip_.empty()) {
        local_address.sin_family = AF_INET;
        if (inet_pton(AF_INET, local_source_ip_.c_str(), &local_address.sin_addr) != 1) {
            result->error = "configured local source IP is not valid IPv4";
            return false;
        }
    }
    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    addrinfo * addresses = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses) != 0) {
        result->error = "HTTP host resolution failed";
        return false;
    }
    if (socket_fd_ < 0) {
        for (addrinfo * address = addresses; address != nullptr; address = address->ai_next) {
            if (!local_source_ip_.empty() && address->ai_family != AF_INET) continue;
            socket_fd_ = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
            if (socket_fd_ >= 0 && !local_source_ip_.empty() &&
                bind(socket_fd_, reinterpret_cast<const sockaddr *>(&local_address),
                    sizeof(local_address)) != 0) {
                close(socket_fd_);
                socket_fd_ = -1;
                continue;
            }
            if (socket_fd_ >= 0 && connect(socket_fd_, address->ai_addr, address->ai_addrlen) == 0) break;
            if (socket_fd_ >= 0) close(socket_fd_);
            socket_fd_ = -1;
        }
    }
    freeaddrinfo(addresses);
    if (socket_fd_ < 0) {
        result->error = "HTTP connection failed";
        return false;
    }
    char local_host[NI_MAXHOST] = {};
    char local_service[NI_MAXSERV] = {};
    char remote_host[NI_MAXHOST] = {};
    char remote_service[NI_MAXSERV] = {};
    sockaddr_storage socket_address{};
    socklen_t socket_address_length = sizeof(socket_address);
    if (getsockname(socket_fd_, reinterpret_cast<sockaddr *>(&socket_address),
        &socket_address_length) == 0 && getnameinfo(
            reinterpret_cast<sockaddr *>(&socket_address), socket_address_length,
            local_host, sizeof(local_host), local_service, sizeof(local_service),
            NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
        result->local_endpoint = std::string(local_host) + ":" + local_service;
    }
    socket_address = {};
    socket_address_length = sizeof(socket_address);
    if (getpeername(socket_fd_, reinterpret_cast<sockaddr *>(&socket_address),
        &socket_address_length) == 0 && getnameinfo(
            reinterpret_cast<sockaddr *>(&socket_address), socket_address_length,
            remote_host, sizeof(remote_host), remote_service, sizeof(remote_service),
            NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
        result->remote_endpoint = std::string(remote_host) + ":" + remote_service;
    }
    const std::string request = "GET " + path + " HTTP/1.1\r\nHost: " + host +
        "\r\nRange: bytes=" + std::to_string(offset) + "-" + std::to_string(end - 1) +
        "\r\nConnection: keep-alive\r\n\r\n";
    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t count = send(socket_fd_, request.data() + sent, request.size() - sent, 0);
        if (count <= 0) { close(socket_fd_); socket_fd_ = -1; result->error = "HTTP request send failed"; return false; }
        sent += static_cast<size_t>(count);
    }
    const uint64_t send_complete_ns = now_ns();
    std::vector<uint8_t> header_bytes;
    size_t body_start = 0;
    while (body_start == 0) {
        uint8_t buffer[4096];
        const ssize_t count = recv(socket_fd_, buffer, sizeof(buffer), 0);
        if (count <= 0 || header_bytes.size() > 65536) {
            close(socket_fd_); socket_fd_ = -1; result->error = "HTTP response headers truncated"; return false;
        }
        header_bytes.insert(header_bytes.end(), buffer, buffer + count);
        const char header_end[] = "\r\n\r\n";
        const auto begin = std::search(header_bytes.begin(), header_bytes.end(),
            header_end, header_end + 4);
        if (begin != header_bytes.end()) body_start = static_cast<size_t>(begin - header_bytes.begin()) + 4;
    }
    const std::string headers(header_bytes.begin(), header_bytes.begin() + body_start);
    std::istringstream header_stream(headers);
    std::string status_line;
    std::getline(header_stream, status_line);
    std::istringstream status_fields(status_line);
    std::string protocol;
    status_fields >> protocol >> result->status_code;
    std::string content_range;
    uint64_t content_length = 0;
    bool have_content_length = false;
    std::string line;
    while (std::getline(header_stream, line)) {
        if (line.size() >= 15 && line.compare(0, 15, "Content-Length:") == 0) {
            content_length = std::stoull(line.substr(15));
            have_content_length = true;
        } else if (line.size() >= 14 && line.compare(0, 14, "Content-Range:") == 0) {
            content_range = line.substr(14);
        }
    }
    result->content_range = content_range;
    if (result->status_code != 206) {
        close(socket_fd_); socket_fd_ = -1; result->error = "HTTP server did not return 206 Partial Content"; return false;
    }
    if (have_content_length && content_length != length) {
        close(socket_fd_); socket_fd_ = -1; result->error = "HTTP returned an incorrect Content-Length"; return false;
    }
    const size_t initial_body_bytes = header_bytes.size() - body_start;
    if (initial_body_bytes > length) {
        close(socket_fd_); socket_fd_ = -1; result->error = "HTTP returned more than the requested range"; return false;
    }
    size_t received = initial_body_bytes;
    uint64_t memcpy_calls = 0;
    uint64_t memcpy_bytes = 0;
    uint64_t memcpy_ns = 0;
    if (initial_body_bytes != 0) {
        if (result->first_byte_timestamp_ns == 0) result->first_byte_timestamp_ns = now_ns();
        const uint64_t copy_start_ns = now_ns();
        std::memcpy(destination, header_bytes.data() + body_start, initial_body_bytes);
        memcpy_ns += now_ns() - copy_start_ns;
        ++memcpy_calls;
        memcpy_bytes += initial_body_bytes;
    }
    while (received < length) {
        uint8_t buffer[4096];
        const size_t remaining = length - received;
        const ssize_t count = recv(socket_fd_, buffer,
            std::min(sizeof(buffer), remaining + 1), 0);
        if (count <= 0) break;
        if (static_cast<size_t>(count) > remaining) {
            close(socket_fd_); socket_fd_ = -1; result->error = "HTTP returned more than the requested range"; return false;
        }
        if (result->first_byte_timestamp_ns == 0) result->first_byte_timestamp_ns = now_ns();
        const uint64_t copy_start_ns = now_ns();
        std::memcpy(destination + received, buffer, static_cast<size_t>(count));
        memcpy_ns += now_ns() - copy_start_ns;
        ++memcpy_calls;
        memcpy_bytes += static_cast<size_t>(count);
        received += static_cast<size_t>(count);
    }
    const uint64_t body_complete_ns = now_ns();
    result->returned_bytes = received;
    if (received != length) {
        close(socket_fd_);
        socket_fd_ = -1;
        result->error = "HTTP returned a truncated or oversized payload";
        return false;
    }
    if (!content_range.empty()) {
        std::istringstream fields(content_range);
        std::string unit;
        uint64_t start = 0;
        char dash = 0;
        uint64_t last = 0;
        fields >> unit >> start >> dash >> last;
        if (fields.fail() || unit.find("bytes") == std::string::npos ||
            start != offset || last != end - 1) {
            close(socket_fd_);
            socket_fd_ = -1;
            result->error = "HTTP Content-Range does not match the request";
            return false;
        }
    }
    vbuf_d0_1_http_request(request_start_ns, send_complete_ns,
        result->first_byte_timestamp_ns, body_complete_ns, length,
        mutex_acquired_ns - request_start_ns, now_ns() - mutex_acquired_ns,
        memcpy_calls, memcpy_bytes, memcpy_ns);
    request_durations_ns_.push_back(now_ns() - request_start_ns);
    transferred_bytes_ += length;
    unique_ranges_.insert(std::to_string(offset) + ":" + std::to_string(length));
    if (!result->local_endpoint.empty()) connection_endpoints_.insert(result->local_endpoint);
    if (result->first_byte_timestamp_ns == 0) result->first_byte_timestamp_ns = now_ns();
    return true;
}

RangeSourceMetrics HttpRangeSource::metrics() const {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    return summarize_metrics(request_durations_ns_, transferred_bytes_, unique_ranges_, connection_endpoints_);
}

ProgressiveRangeSource::ProgressiveRangeSource(std::shared_ptr<RangeSource> remote,
    std::string mirror_path, ProgressiveSourceIdentity identity)
    : remote_(std::move(remote)), mirror_path_(std::move(mirror_path)),
      coverage_path_(mirror_path_ + ".coverage"), identity_(identity) {
    if (!remote_ || identity_.hash_algorithm != 1 || identity_.declared_size == 0) {
        throw std::runtime_error("invalid progressive source identity");
    }
    if (!open_store()) {
        if (coverage_fd_ >= 0) close(coverage_fd_);
        if (mirror_fd_ >= 0) close(mirror_fd_);
        coverage_fd_ = -1;
        mirror_fd_ = -1;
        throw std::runtime_error("progressive source store cannot be opened");
    }
}

ProgressiveRangeSource::~ProgressiveRangeSource() {
    if (coverage_fd_ >= 0) close(coverage_fd_);
    if (mirror_fd_ >= 0) close(mirror_fd_);
}

uint64_t ProgressiveRangeSource::chunk_count() const {
    return identity_.declared_size / CHUNK_SIZE +
        (identity_.declared_size % CHUNK_SIZE == 0 ? 0 : 1);
}

uint64_t ProgressiveRangeSource::chunk_length(uint64_t chunk_index) const {
    const uint64_t offset = chunk_index * CHUNK_SIZE;
    return std::min(CHUNK_SIZE, identity_.declared_size - offset);
}

bool ProgressiveRangeSource::valid_range(uint64_t offset, uint64_t length, uint64_t * end) const {
    return length != 0 && checked_end(offset, length, end) && *end <= identity_.declared_size;
}

bool ProgressiveRangeSource::read_at(uint64_t offset, uint8_t * destination, size_t length) const {
    return positioned_read(mirror_fd_, destination, length, offset);
}

bool ProgressiveRangeSource::write_at(uint64_t offset, const uint8_t * source, size_t length) {
    return positioned_write(mirror_fd_, source, length, offset);
}

bool ProgressiveRangeSource::read_local(uint64_t offset, uint64_t length, uint8_t * destination) {
    return read_at(offset, destination, static_cast<size_t>(length));
}

bool ProgressiveRangeSource::bit_is_set(uint64_t chunk_index) const {
    const size_t byte = static_cast<size_t>(chunk_index / 8);
    const uint8_t mask = static_cast<uint8_t>(1u << (chunk_index % 8));
    return (coverage_[byte] & mask) != 0;
}

void ProgressiveRangeSource::set_bit(uint64_t chunk_index) {
    const size_t byte = static_cast<size_t>(chunk_index / 8);
    coverage_[byte] |= static_cast<uint8_t>(1u << (chunk_index % 8));
}

bool ProgressiveRangeSource::publish_chunk(uint64_t chunk_index) {
    const size_t byte = static_cast<size_t>(chunk_index / 8);
    const uint8_t mask = static_cast<uint8_t>(1u << (chunk_index % 8));
    const uint8_t value = static_cast<uint8_t>(coverage_[byte] | mask);
    if (!positioned_write(coverage_fd_, &value, 1, PROGRESSIVE_COVERAGE_HEADER_BYTES + byte)) {
        return false;
    }
    if (!bit_is_set(chunk_index)) {
        set_bit(chunk_index);
        ++metrics_.covered_chunks;
        ++metrics_.chunks_published;
    }
    return true;
}

bool ProgressiveRangeSource::ensure_chunk(uint64_t chunk_index) {
    if (bit_is_set(chunk_index)) {
        ++metrics_.local_chunk_hits;
        return true;
    }
    const uint64_t offset = chunk_index * CHUNK_SIZE;
    const uint64_t length = chunk_length(chunk_index);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    RangeReadResult remote_result;
    ++metrics_.remote_chunk_misses;
    if (!remote_->read_range(offset, length, bytes.data(), &remote_result) ||
        remote_result.requested_offset != offset || remote_result.requested_length != length ||
        remote_result.returned_bytes != length) {
        return false;
    }
    metrics_.remote_source_bytes += length;
    if (!write_at(offset, bytes.data(), bytes.size())) return false;
    return publish_chunk(chunk_index);
}

bool ProgressiveRangeSource::read_range(uint64_t offset, uint64_t length,
    uint8_t * destination, RangeReadResult * result) {
    RangeReadResult local_result;
    if (result == nullptr) result = &local_result;
    *result = {};
    result->requested_offset = offset;
    result->requested_length = length;
    result->source_id = "progressive-local";
    uint64_t end = 0;
    if (destination == nullptr || !valid_range(offset, length, &end)) {
        result->error = "progressive range is outside the declared source";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t first_chunk = offset / CHUNK_SIZE;
    const uint64_t last_chunk = (end - 1) / CHUNK_SIZE;
    for (uint64_t chunk = first_chunk; chunk <= last_chunk; ++chunk) {
        if (!ensure_chunk(chunk)) {
            result->error = "progressive source acquisition or publication failed";
            return false;
        }
    }
    if (!read_local(offset, length, destination)) {
        result->error = "progressive local mirror read failed";
        return false;
    }
    metrics_.local_source_bytes += length;
    result->first_byte_timestamp_ns = now_ns();
    result->returned_bytes = length;
    result->status_code = 200;
    return true;
}

RangeSourceMetrics ProgressiveRangeSource::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    RangeSourceMetrics result{};
    result.requests = metrics_.remote_chunk_misses;
    result.bytes = metrics_.remote_source_bytes;
    result.unique_bytes = metrics_.remote_source_bytes;
    return result;
}

ProgressiveRangeMetrics ProgressiveRangeSource::progressive_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

bool ProgressiveRangeSource::chunk_is_covered(uint64_t chunk_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return chunk_index < chunk_count() && bit_is_set(chunk_index);
}

bool ProgressiveRangeSource::write_sidecar_header() const {
    std::array<uint8_t, PROGRESSIVE_COVERAGE_HEADER_BYTES> header{};
    std::copy(PROGRESSIVE_COVERAGE_MAGIC.begin(), PROGRESSIVE_COVERAGE_MAGIC.end(), header.begin());
    put_u32(header.data() + 8, PROGRESSIVE_COVERAGE_VERSION);
    put_u64(header.data() + 16, identity_.declared_size);
    put_u16(header.data() + 24, identity_.hash_algorithm);
    put_u16(header.data() + 26, static_cast<uint16_t>(identity_.full_source_hash.size()));
    put_u32(header.data() + 28, static_cast<uint32_t>(CHUNK_SIZE));
    put_u64(header.data() + 32, chunk_count());
    put_u64(header.data() + 40, coverage_.size());
    std::copy(identity_.full_source_hash.begin(), identity_.full_source_hash.end(), header.begin() + 48);
    return positioned_write(coverage_fd_, header.data(), header.size(), 0);
}

bool ProgressiveRangeSource::initialize_sidecar() {
    coverage_.assign(static_cast<size_t>((chunk_count() + 7) / 8), 0);
    metrics_.coverage_bytes = coverage_.size();
    if (!write_sidecar_header()) return false;
    if (!coverage_.empty() && !positioned_write(coverage_fd_, coverage_.data(), coverage_.size(),
        PROGRESSIVE_COVERAGE_HEADER_BYTES)) return false;
    return true;
}

bool ProgressiveRangeSource::read_sidecar_header() {
    std::array<uint8_t, PROGRESSIVE_COVERAGE_HEADER_BYTES> header{};
    if (!positioned_read(coverage_fd_, header.data(), header.size(), 0) ||
        !std::equal(PROGRESSIVE_COVERAGE_MAGIC.begin(), PROGRESSIVE_COVERAGE_MAGIC.end(), header.begin()) ||
        get_u32(header.data() + 8) != PROGRESSIVE_COVERAGE_VERSION ||
        get_u64(header.data() + 16) != identity_.declared_size ||
        get_u16(header.data() + 24) != identity_.hash_algorithm ||
        get_u16(header.data() + 26) != identity_.full_source_hash.size() ||
        get_u32(header.data() + 28) != CHUNK_SIZE ||
        get_u64(header.data() + 32) != chunk_count() ||
        get_u64(header.data() + 40) != (chunk_count() + 7) / 8 ||
        !std::equal(identity_.full_source_hash.begin(), identity_.full_source_hash.end(), header.begin() + 48)) {
        return false;
    }
    coverage_.resize(static_cast<size_t>((chunk_count() + 7) / 8));
    metrics_.coverage_bytes = coverage_.size();
    if (!coverage_.empty() && !positioned_read(coverage_fd_, coverage_.data(), coverage_.size(),
        PROGRESSIVE_COVERAGE_HEADER_BYTES)) return false;
    if (chunk_count() % 8 != 0 && !coverage_.empty()) {
        const uint8_t valid_mask = static_cast<uint8_t>((1u << (chunk_count() % 8)) - 1u);
        if ((coverage_.back() & static_cast<uint8_t>(~valid_mask)) != 0) return false;
    }
    metrics_.covered_chunks = 0;
    for (uint64_t chunk = 0; chunk < chunk_count(); ++chunk)
        if (bit_is_set(chunk)) ++metrics_.covered_chunks;
    return true;
}

bool ProgressiveRangeSource::open_store() {
    mirror_fd_ = open(mirror_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (mirror_fd_ < 0) return false;
    struct stat mirror_stat{};
    if (fstat(mirror_fd_, &mirror_stat) != 0) return false;
    coverage_fd_ = open(coverage_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (coverage_fd_ < 0) return false;
    struct stat coverage_stat{};
    if (fstat(coverage_fd_, &coverage_stat) != 0) return false;
    if (mirror_stat.st_size == 0 && coverage_stat.st_size != 0) return false;
    if (mirror_stat.st_size == 0) {
        if (ftruncate(mirror_fd_, static_cast<off_t>(identity_.declared_size)) != 0) return false;
    } else if (static_cast<uint64_t>(mirror_stat.st_size) != identity_.declared_size) {
        return false;
    }
    if (coverage_stat.st_size == 0) return initialize_sidecar();
    const uint64_t expected_size = PROGRESSIVE_COVERAGE_HEADER_BYTES + (chunk_count() + 7) / 8;
    if (static_cast<uint64_t>(coverage_stat.st_size) != expected_size) return false;
    return read_sidecar_header();
}

} // namespace vbuf_ggml

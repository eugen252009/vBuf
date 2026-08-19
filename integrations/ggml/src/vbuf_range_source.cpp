#include "vbuf_range_source.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
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

} // namespace

LocalVbufRangeSource::LocalVbufRangeSource(const uint8_t * mapped_base,
    uint64_t artifact_bytes, std::string source_id)
    : mapped_base_(mapped_base), artifact_bytes_(artifact_bytes), source_id_(std::move(source_id)) {}

bool LocalVbufRangeSource::read_range(uint64_t offset, uint64_t length,
    uint8_t * destination, RangeReadResult * result) {
    *result = { offset, length, 0, 0, 0, source_id_, {}, {} };
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
    std::lock_guard<std::mutex> socket_lock(socket_mutex_);
    const uint64_t request_start_ns = now_ns();
    *result = { offset, length, 0, 0, 0, endpoint_, {}, {} };
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
    if (initial_body_bytes != 0) {
        if (result->first_byte_timestamp_ns == 0) result->first_byte_timestamp_ns = now_ns();
        std::memcpy(destination, header_bytes.data() + body_start, initial_body_bytes);
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
        std::memcpy(destination + received, buffer, static_cast<size_t>(count));
        received += static_cast<size_t>(count);
    }
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

} // namespace vbuf_ggml

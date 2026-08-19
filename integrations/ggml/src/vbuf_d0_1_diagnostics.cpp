#include "vbuf_d0_1_diagnostics.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#include <sys/system_properties.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

bool property_enabled(const char * name) {
#if defined(__ANDROID__)
    char value[PROP_VALUE_MAX] = {};
    return __system_property_get(name, value) > 0 && value[0] == '1';
#else
    (void) name;
    return false;
#endif
}

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}

uint64_t elapsed_ns(uint64_t start, uint64_t end) {
    return end >= start ? end - start : 0;
}

uint64_t percentile(std::vector<uint64_t> values, double fraction) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>((values.size() - 1) * fraction);
    return values[index];
}

struct Snapshot {
    std::string phase;
    uint64_t rss_kib = 0;
    uint64_t pss_kib = 0;
    uint64_t swap_pss_kib = 0;
};

struct Session {
    std::mutex mutex;
    bool active = false;
    bool enabled = false;
    bool smaps_enabled = false;
    uint64_t model_open_begin_ns = 0;
    uint64_t bootstrap_begin_ns = 0;
    uint64_t bootstrap_complete_ns = 0;
    uint64_t descriptor_begin_ns = 0;
    uint64_t descriptor_complete_ns = 0;
    uint64_t first_payload_request_ns = 0;
    uint64_t final_payload_ready_ns = 0;
    uint64_t model_create_begin_ns = 0;
    uint64_t model_create_complete_ns = 0;
    uint64_t backend_allocation_begin_ns = 0;
    uint64_t backend_allocation_complete_ns = 0;
    uint64_t pointer_bind_begin_ns = 0;
    uint64_t pointer_bind_complete_ns = 0;
    uint64_t model_open_complete_ns = 0;
    uint64_t post_generation_ns = 0;
    uint64_t tensor_count = 0;
    uint64_t ready_count = 0;

    uint64_t request_count = 0;
    uint64_t request_setup_total_ns = 0;
    uint64_t worker_create_total_ns = 0;
    uint64_t worker_start_delay_total_ns = 0;
    uint64_t worker_wait_total_ns = 0;
    uint64_t request_finalization_total_ns = 0;
    uint64_t http_bytes = 0;
    uint64_t http_header_setup_total_ns = 0;
    uint64_t memcpy_calls = 0;
    uint64_t memcpy_bytes = 0;
    uint64_t memcpy_total_ns = 0;
    uint64_t hash_calls = 0;
    uint64_t hash_bytes = 0;
    uint64_t hash_total_ns = 0;
    uint64_t smaps_calls = 0;
    uint64_t smaps_total_ns = 0;
    uint64_t callback_bind_count = 0;
    uint64_t callback_lookup_iterations = 0;
    uint64_t callback_lookup_total_ns = 0;
    uint64_t pointer_bind_total_ns = 0;
    uint64_t backend_allocation_count = 0;
    uint64_t backend_allocated_bytes = 0;
    uint64_t backend_allocation_total_ns = 0;
    uint64_t http_mutex_wait_total_ns = 0;
    uint64_t http_mutex_held_total_ns = 0;
    std::vector<uint64_t> request_to_first_byte_ns;
    std::vector<uint64_t> body_receive_ns;
    std::vector<uint64_t> smaps_durations_ns;
    std::vector<Snapshot> snapshots;

    void reset() {
        const bool requested = std::getenv("VBUF_D0_1_TRACE") != nullptr || property_enabled("debug.vbuf.d0_1.trace");
        const bool disable_smaps = std::getenv("VBUF_D0_1_DISABLE_SMAPS") != nullptr || property_enabled("debug.vbuf.d0_1.disable_smaps");
        model_open_begin_ns = bootstrap_begin_ns = bootstrap_complete_ns = 0;
        descriptor_begin_ns = descriptor_complete_ns = first_payload_request_ns = 0;
        final_payload_ready_ns = model_create_begin_ns = model_create_complete_ns = 0;
        backend_allocation_begin_ns = backend_allocation_complete_ns = 0;
        pointer_bind_begin_ns = pointer_bind_complete_ns = model_open_complete_ns = 0;
        post_generation_ns = tensor_count = ready_count = 0;
        request_count = request_setup_total_ns = worker_create_total_ns = 0;
        worker_start_delay_total_ns = worker_wait_total_ns = request_finalization_total_ns = 0;
        http_bytes = http_header_setup_total_ns = memcpy_calls = memcpy_bytes = 0;
        memcpy_total_ns = hash_calls = hash_bytes = hash_total_ns = 0;
        smaps_calls = smaps_total_ns = callback_bind_count = callback_lookup_iterations = 0;
        callback_lookup_total_ns = pointer_bind_total_ns = backend_allocation_count = 0;
        backend_allocated_bytes = backend_allocation_total_ns = 0;
        http_mutex_wait_total_ns = http_mutex_held_total_ns = 0;
        request_to_first_byte_ns.clear();
        body_receive_ns.clear();
        smaps_durations_ns.clear();
        snapshots.clear();
        enabled = requested;
        smaps_enabled = enabled && !disable_smaps;
        active = enabled;
        if (active) model_open_begin_ns = now_ns();
    }
};

Session session;

bool read_snapshot(Snapshot & snapshot) {
    std::ifstream input("/proc/self/smaps_rollup");
    if (input) {
        std::string key;
        uint64_t value = 0;
        while (input >> key >> value) {
            if (key == "Rss:") snapshot.rss_kib = value;
            else if (key == "Pss:") snapshot.pss_kib = value;
            else if (key == "SwapPss:") snapshot.swap_pss_kib = value;
            std::string rest;
            std::getline(input, rest);
        }
    }
    if (snapshot.rss_kib == 0) {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            std::istringstream fields(line);
            std::string key;
            uint64_t value = 0;
            fields >> key >> value;
            if (key == "VmRSS:") { snapshot.rss_kib = value; break; }
        }
    }
    return snapshot.rss_kib != 0 || snapshot.pss_kib != 0;
}

void snapshot_locked(const char * phase) {
    if (!session.enabled) return;
    Snapshot snapshot;
    snapshot.phase = phase;
    if (read_snapshot(snapshot)) session.snapshots.push_back(std::move(snapshot));
}

uint64_t duration_from(uint64_t start, uint64_t end) {
    return start == 0 ? 0 : elapsed_ns(start, end);
}

uint64_t ms(uint64_t ns) { return ns / 1000000; }

void log_line(const char * format, ...) {
    char line[16384] = {};
    va_list args;
    va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    std::fprintf(stderr, "%s", line);
#if defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_INFO, "vbuf-d0-1", "%s", line);
#endif
}

void emit_locked() {
    if (!session.enabled) return;
    const uint64_t total_ns = duration_from(session.model_open_begin_ns, session.model_open_complete_ns);
    const uint64_t serialized_ns = duration_from(session.first_payload_request_ns, session.final_payload_ready_ns);
    const uint64_t post_payload_ns = duration_from(session.final_payload_ready_ns, session.model_open_complete_ns);
    log_line("VBUF_D0_1_JSON {\"model_open_total_ns\":%llu,\"bootstrap_ns\":%llu,\"descriptor_loop_ns\":%llu,\"model_create_ns\":%llu,\"serialized_payload_ns\":%llu,\"post_final_payload_ns\":%llu,\"backend_allocation_ns\":%llu,\"pointer_bind_ns\":%llu,\"request_count\":%llu}\n",
        static_cast<unsigned long long>(total_ns), static_cast<unsigned long long>(duration_from(session.bootstrap_begin_ns, session.bootstrap_complete_ns)),
        static_cast<unsigned long long>(duration_from(session.descriptor_begin_ns, session.descriptor_complete_ns)), static_cast<unsigned long long>(duration_from(session.model_create_begin_ns, session.model_create_complete_ns)),
        static_cast<unsigned long long>(serialized_ns), static_cast<unsigned long long>(post_payload_ns), static_cast<unsigned long long>(duration_from(session.backend_allocation_begin_ns, session.backend_allocation_complete_ns)),
        static_cast<unsigned long long>(duration_from(session.pointer_bind_begin_ns, session.pointer_bind_complete_ns)), static_cast<unsigned long long>(session.request_count));
    log_line("VBUF_D0_1_HTTP {\"request_to_first_byte_total_ns\":%llu,\"request_to_first_byte_median_ns\":%llu,\"request_to_first_byte_p95_ns\":%llu,\"body_receive_total_ns\":%llu,\"body_receive_median_ns\":%llu,\"body_receive_p95_ns\":%llu,\"http_bytes\":%llu,\"request_setup_total_ns\":%llu,\"worker_create_total_ns\":%llu,\"worker_wait_total_ns\":%llu,\"worker_start_delay_total_ns\":%llu,\"request_finalization_total_ns\":%llu,\"http_mutex_wait_total_ns\":%llu,\"http_mutex_held_total_ns\":%llu,\"http_header_setup_total_ns\":%llu}\n",
        static_cast<unsigned long long>(std::accumulate(session.request_to_first_byte_ns.begin(), session.request_to_first_byte_ns.end(), uint64_t{0})),
         static_cast<unsigned long long>(percentile(session.request_to_first_byte_ns, 0.50)), static_cast<unsigned long long>(percentile(session.request_to_first_byte_ns, 0.95)),
        static_cast<unsigned long long>(std::accumulate(session.body_receive_ns.begin(), session.body_receive_ns.end(), uint64_t{0})),
         static_cast<unsigned long long>(percentile(session.body_receive_ns, 0.50)), static_cast<unsigned long long>(percentile(session.body_receive_ns, 0.95)),
        static_cast<unsigned long long>(session.http_bytes), static_cast<unsigned long long>(session.request_setup_total_ns),
        static_cast<unsigned long long>(session.worker_create_total_ns), static_cast<unsigned long long>(session.worker_wait_total_ns), static_cast<unsigned long long>(session.worker_start_delay_total_ns),
        static_cast<unsigned long long>(session.request_finalization_total_ns), static_cast<unsigned long long>(session.http_mutex_wait_total_ns), static_cast<unsigned long long>(session.http_mutex_held_total_ns), static_cast<unsigned long long>(session.http_header_setup_total_ns));
    log_line("VBUF_D0_1_CPU {\"memcpy_calls\":%llu,\"memcpy_bytes\":%llu,\"memcpy_total_ns\":%llu,\"hash_calls\":%llu,\"hash_bytes\":%llu,\"hash_total_ns\":%llu,\"smaps_calls\":%llu,\"smaps_total_ns\":%llu,\"smaps_median_ns\":%llu,\"smaps_p95_ns\":%llu}\n",
        static_cast<unsigned long long>(session.memcpy_calls), static_cast<unsigned long long>(session.memcpy_bytes), static_cast<unsigned long long>(session.memcpy_total_ns),
        static_cast<unsigned long long>(session.hash_calls), static_cast<unsigned long long>(session.hash_bytes), static_cast<unsigned long long>(session.hash_total_ns),
        static_cast<unsigned long long>(session.smaps_calls), static_cast<unsigned long long>(session.smaps_total_ns),
        static_cast<unsigned long long>(percentile(session.smaps_durations_ns, 0.50)), static_cast<unsigned long long>(percentile(session.smaps_durations_ns, 0.95)));
    for (const Snapshot & snapshot : session.snapshots) {
        log_line("VBUF_D0_1_SNAPSHOT {\"phase\":\"%s\",\"rss_kib\":%llu,\"pss_kib\":%llu,\"swap_pss_kib\":%llu}\n",
            snapshot.phase.c_str(), static_cast<unsigned long long>(snapshot.rss_kib),
            static_cast<unsigned long long>(snapshot.pss_kib), static_cast<unsigned long long>(snapshot.swap_pss_kib));
    }
}

} // namespace

extern "C" void vbuf_d0_1_begin() {
        std::lock_guard<std::mutex> lock(session.mutex);
    if (!session.active || session.model_open_complete_ns != 0) session.reset();
}

extern "C" void vbuf_d0_1_bootstrap_begin() { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.bootstrap_begin_ns = now_ns(); }
extern "C" void vbuf_d0_1_bootstrap_complete() { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { session.bootstrap_complete_ns = now_ns(); snapshot_locked("BOOTSTRAP_COMPLETE"); } }
extern "C" void vbuf_d0_1_descriptor_loop_begin() { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.descriptor_begin_ns = now_ns(); }
extern "C" void vbuf_d0_1_descriptor_loop_complete() { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.descriptor_complete_ns = now_ns(); }
extern "C" void vbuf_d0_1_set_tensor_count(uint64_t count) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.tensor_count = count; }
extern "C" void vbuf_d0_1_payload_ready(uint64_t ready_count, uint64_t timestamp_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { session.ready_count = ready_count; if (session.tensor_count != 0 && ready_count * 2 == session.tensor_count) snapshot_locked("HALF_PAYLOADS_READY"); } }
extern "C" void vbuf_d0_1_first_payload_request(uint64_t timestamp_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active && session.first_payload_request_ns == 0) session.first_payload_request_ns = timestamp_ns; }
extern "C" void vbuf_d0_1_final_payload_ready(uint64_t timestamp_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active && session.final_payload_ready_ns == 0) { session.final_payload_ready_ns = timestamp_ns; snapshot_locked("FINAL_PAYLOAD_READY"); } }
extern "C" void vbuf_d0_1_model_create(bool begin) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { if (begin) session.model_create_begin_ns = now_ns(); else session.model_create_complete_ns = now_ns(); } }
extern "C" void vbuf_d0_1_backend_allocation(bool begin, const char *, uint64_t bytes) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { if (begin) session.backend_allocation_begin_ns = now_ns(); else { session.backend_allocation_complete_ns = now_ns(); ++session.backend_allocation_count; session.backend_allocated_bytes += bytes; session.backend_allocation_total_ns += duration_from(session.backend_allocation_begin_ns, session.backend_allocation_complete_ns); snapshot_locked("BACKEND_ALLOCATION_COMPLETE"); } } }
extern "C" void vbuf_d0_1_pointer_bind_begin() { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { ++session.callback_bind_count; session.pointer_bind_begin_ns = now_ns(); } }
extern "C" void vbuf_d0_1_pointer_bind_complete(uint64_t iterations, uint64_t lookup_ns, uint64_t pointer_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { session.callback_lookup_iterations += iterations; session.callback_lookup_total_ns += lookup_ns; session.pointer_bind_total_ns += pointer_ns; session.pointer_bind_complete_ns = now_ns(); } }
extern "C" void vbuf_d0_1_request_setup(uint64_t duration_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.request_setup_total_ns += duration_ns; }
extern "C" void vbuf_d0_1_worker_create(uint64_t duration_ns, uint64_t start_delay_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { session.worker_create_total_ns += duration_ns; session.worker_start_delay_total_ns += start_delay_ns; } }
extern "C" void vbuf_d0_1_worker_start_delay(uint64_t duration_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.worker_start_delay_total_ns += duration_ns; }
extern "C" void vbuf_d0_1_worker_wait(uint64_t duration_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.worker_wait_total_ns += duration_ns; }
extern "C" void vbuf_d0_1_request_finalization(uint64_t duration_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) session.request_finalization_total_ns += duration_ns; }
extern "C" void vbuf_d0_1_http_request(uint64_t request_start_ns, uint64_t send_complete_ns, uint64_t first_body_byte_ns, uint64_t body_complete_ns, uint64_t bytes, uint64_t mutex_wait_ns, uint64_t mutex_held_ns, uint64_t memcpy_calls, uint64_t memcpy_bytes, uint64_t memcpy_ns) {
    std::lock_guard<std::mutex> lock(session.mutex);
    if (!session.active) return;
    ++session.request_count;
    session.http_bytes += bytes;
    session.http_mutex_wait_total_ns += mutex_wait_ns;
    session.http_mutex_held_total_ns += mutex_held_ns;
    session.http_header_setup_total_ns += elapsed_ns(send_complete_ns, first_body_byte_ns);
    session.memcpy_calls += memcpy_calls;
    session.memcpy_bytes += memcpy_bytes;
    session.memcpy_total_ns += memcpy_ns;
    session.request_to_first_byte_ns.push_back(elapsed_ns(request_start_ns, first_body_byte_ns));
    session.body_receive_ns.push_back(elapsed_ns(first_body_byte_ns, body_complete_ns));
}
extern "C" void vbuf_d0_1_hash(uint64_t bytes, uint64_t duration_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { ++session.hash_calls; session.hash_bytes += bytes; session.hash_total_ns += duration_ns; } }
extern "C" void vbuf_d0_1_smaps(uint64_t duration_ns) { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { ++session.smaps_calls; session.smaps_total_ns += duration_ns; session.smaps_durations_ns.push_back(duration_ns); } }
extern "C" bool vbuf_d0_1_smaps_enabled() { std::lock_guard<std::mutex> lock(session.mutex); return session.smaps_enabled; }
extern "C" void vbuf_d0_1_model_open_complete() { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { session.model_open_complete_ns = now_ns(); snapshot_locked("MODEL_OPEN_COMPLETE"); emit_locked(); } }
extern "C" void vbuf_d0_1_post_generation() { std::lock_guard<std::mutex> lock(session.mutex); if (session.active) { session.post_generation_ns = now_ns(); snapshot_locked("POST_GENERATION"); emit_locked(); } }
extern "C" void vbuf_d0_1_emit_report() { std::lock_guard<std::mutex> lock(session.mutex); emit_locked(); }

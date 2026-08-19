#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void vbuf_d0_1_begin();
void vbuf_d0_1_bootstrap_begin();
void vbuf_d0_1_bootstrap_complete();
void vbuf_d0_1_descriptor_loop_begin();
void vbuf_d0_1_descriptor_loop_complete();
void vbuf_d0_1_set_tensor_count(uint64_t count);
void vbuf_d0_1_payload_ready(uint64_t ready_count, uint64_t timestamp_ns);
void vbuf_d0_1_first_payload_request(uint64_t timestamp_ns);
void vbuf_d0_1_final_payload_ready(uint64_t timestamp_ns);
void vbuf_d0_1_model_create(bool begin);
void vbuf_d0_1_backend_allocation(bool begin, const char * name, uint64_t bytes);
void vbuf_d0_1_pointer_bind_begin();
void vbuf_d0_1_pointer_bind_complete(uint64_t lookup_iterations,
    uint64_t lookup_ns, uint64_t pointer_bind_ns);
void vbuf_d0_1_request_setup(uint64_t duration_ns);
void vbuf_d0_1_worker_create(uint64_t duration_ns, uint64_t start_delay_ns);
void vbuf_d0_1_worker_start_delay(uint64_t duration_ns);
void vbuf_d0_1_worker_wait(uint64_t duration_ns);
void vbuf_d0_1_request_finalization(uint64_t duration_ns);
void vbuf_d0_1_http_request(uint64_t request_start_ns, uint64_t send_complete_ns,
    uint64_t first_body_byte_ns, uint64_t body_complete_ns, uint64_t bytes,
    uint64_t mutex_wait_ns, uint64_t mutex_held_ns,
    uint64_t memcpy_calls, uint64_t memcpy_bytes, uint64_t memcpy_ns);
void vbuf_d0_1_hash(uint64_t bytes, uint64_t duration_ns);
void vbuf_d0_1_smaps(uint64_t duration_ns);
bool vbuf_d0_1_smaps_enabled();
void vbuf_d0_1_model_open_complete();
void vbuf_d0_1_post_generation();
void vbuf_d0_1_emit_report();

#ifdef __cplusplus
}
#endif

#ifndef VBUF_H
#define VBUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAGIC 0x46554256   // "VBUF"
#define VERSION 0x00050000 // 0.5.0

typedef struct {
  uint8_t *mem;
  uint8_t *data;
  size_t size;
  uint32_t alignment;
} vbuf_instance_t;

typedef enum { VBUF_U16 = 16, VBUF_U32 = 32, VBUF_U64 = 64 } vbuf_type_t;

/* Canonical v0.6 reader API. The structure is opaque and created only after
 * complete v0.6 validation. Return code 0 means success; other values are
 * stable error categories defined by the Rust reader. */
typedef struct vbuf_v06_instance vbuf_v06_instance_t;
typedef struct vbuf_v06_writer vbuf_v06_writer_t;
typedef enum {
  VBUF_V06_U8 = 0,
  VBUF_V06_U16 = 1,
  VBUF_V06_U32 = 2,
  VBUF_V06_U64 = 3,
  VBUF_V06_I8 = 4,
  VBUF_V06_I16 = 5,
  VBUF_V06_I32 = 6,
  VBUF_V06_I64 = 7,
  VBUF_V06_F32 = 8,
  VBUF_V06_F64 = 9,
  VBUF_V06_OPAQUE = 10
} vbuf_v06_type_t;

vbuf_v06_instance_t *vbuf_v06_open(const char *path, uint32_t *error_out);
void vbuf_v06_close(vbuf_v06_instance_t *instance);
uint32_t vbuf_v06_get(const vbuf_v06_instance_t *instance, uint16_t key_id,
                      size_t occurrence, uint8_t expected_type,
                      const void **data_out, size_t *count_out);
uint32_t vbuf_v06_header_info(const vbuf_v06_instance_t *instance,
                              uint64_t *base_step_out,
                              uint64_t *data_start_out,
                              uint64_t *data_size_out);
uint32_t vbuf_v06_block_info(const vbuf_v06_instance_t *instance, size_t index,
                             uint64_t *block_start_out,
                             uint64_t *payload_start_out,
                             uint64_t *payload_length_out,
                             uint64_t *next_block_start_out,
                             uint64_t *count_out);
/* kind 0 is physical block range; kind 1 is payload range. Returned ranges
 * and alignment are borrowed from the live opaque instance. */
uint32_t vbuf_v06_range_info(const vbuf_v06_instance_t *instance, size_t index,
                             uint8_t kind, uint64_t *offset_out,
                             uint64_t *length_out, uint64_t *end_out,
                             uint64_t *alignment_out);
uint32_t vbuf_v06_range_subrange(const vbuf_v06_instance_t *instance,
                                 size_t index, uint8_t kind,
                                 uint64_t relative_offset, uint64_t length,
                                 uint64_t *offset_out, uint64_t *length_out,
                                 uint64_t *end_out);
/* Returns a borrowed pointer only after the complete selected range is checked;
 * it remains valid while instance is live. */
uint32_t vbuf_v06_range_ptr(const vbuf_v06_instance_t *instance, size_t index,
                            uint8_t kind, const void **data_out,
                            size_t *length_out);
uint32_t vbuf_v06_range_require_alignment(const vbuf_v06_instance_t *instance,
                                           size_t index, uint8_t kind,
                                           uint64_t alignment);

/* Canonical portable v0.6 writer. physical is 0 (scalar) or 1 (array); data
 * points to `count` native C values selected by vbuf_v06_type_t and is encoded
 * little-endian. Writer calls return 0 on success and nonzero on failure.
 * finish always consumes the writer handle. */
vbuf_v06_writer_t *vbuf_v06_writer_create(const char *path, uint8_t base_shift,
                                           bool indefinite,
                                           uint32_t *error_out);
uint32_t vbuf_v06_writer_write(vbuf_v06_writer_t *writer, uint16_t key_id,
                               uint8_t physical, bool continuation,
                               uint8_t payload_shift, uint8_t value_type,
                               const void *data, size_t count);
uint32_t vbuf_v06_writer_finish(vbuf_v06_writer_t *writer);

// --- LEGACY v0.5 CORE API ---
vbuf_instance_t *vbuf_open(const char *filename);
void vbuf_close(vbuf_instance_t *inst);

// --- WRITER API ---
void vbuf_write_header(FILE *f, uint8_t a_shift);
void vbuf_write_atomic_column(FILE *f, uint32_t id, size_t n,
                              uint32_t alignment, uint16_t bit_width,
                              const void *data);
void vbuf_write_column(FILE *f, uint32_t id, size_t n, uint32_t alignment,
                       const uint16_t *data);
size_t vbuf_pack_block(void *target, uint16_t key_id, uint8_t type,
                       const void *data, uint32_t count);


// --- GENERIC DISPATCHER (Neu in vbuf.c implementiert) ---
const void *vbuf_get_col_ptr(vbuf_instance_t *inst, uint32_t id, size_t *n_out,
                             uint16_t *width_out);

// --- LEGACY v0.5 GETTER ---
// This preserves observed pre-v0.6 behavior and is not a safe v0.6 API.
static inline const void *vbuf_get_generic(vbuf_instance_t *inst,
                                           uint32_t key_id, size_t *count_out,
                                           uint16_t *width_out) {
  uint8_t *curr = inst->mem + 16;
  uint8_t *end = inst->mem + inst->size;

  while (curr + 8 <= end) {
    uint64_t anchor = *(uint64_t *)curr;
    uint16_t current_id = (uint16_t)((anchor >> 16) & 0xFFFF);
    uint16_t bit_width = (uint16_t)((anchor >> 32) & 0xFFFF);

    if (anchor == 0) {
      curr += 8;
      continue;
    }

    uint64_t count;
    size_t header_size;
    if (anchor & (1ULL << 9)) {
      if (curr + 16 > end) break;
      count = *(const uint64_t *)(curr + 8);
      header_size = 16;
    } else {
      count = (anchor >> 48) & 0xFFFF;
      header_size = 8;
    }

    size_t current_pos = (size_t)(curr - inst->mem);
    size_t align = (size_t)inst->alignment;

    size_t payload_offset = (current_pos + header_size + (align - 1)) & ~(align - 1);

    if (current_id == (uint16_t)key_id) {
      if (count_out)
        *count_out = (size_t)count;
      if (width_out)
        *width_out = bit_width;
      return (const void *)(inst->mem + payload_offset);
    }

    // Sprung zur nächsten Spalte
    size_t data_bytes = (size_t)count * (bit_width / 8);
    size_t next_off = payload_offset + data_bytes;
    next_off = (next_off + 7) & ~7; // 64-Bit Alignment für den nächsten Anchor

    curr = inst->mem + next_off;
  }
  return NULL;
}

// Komfort-Wrapper für den alten u16-Code
static inline const uint16_t *vbuf_get_u16(vbuf_instance_t *inst,
                                           uint32_t key_id, size_t *count_out) {
  return (const uint16_t *)vbuf_get_generic(inst, key_id, count_out, NULL);
}

#ifdef __cplusplus
}
#endif
#endif

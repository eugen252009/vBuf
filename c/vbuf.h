#ifndef VBUF_H
#define VBUF_H

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

// --- CORE API ---
vbuf_instance_t *vbuf_open(const char *filename);
void vbuf_close(vbuf_instance_t *inst);

// --- WRITER API ---
void vbuf_write_header(FILE *f, uint8_t a_shift);
void vbuf_write_atomic_column(FILE *f, uint32_t id, size_t n,
                              uint32_t alignment, uint16_t bit_width,
                              const void *data);
void vbuf_write_column(FILE *f, uint32_t id, size_t n, uint32_t alignment,
                       const uint16_t *data);

// --- GENERIC DISPATCHER (Neu in vbuf.c implementiert) ---
const void *vbuf_get_col_ptr(vbuf_instance_t *inst, uint32_t id, size_t *n_out,
                             uint16_t *width_out);

// --- HIGH-SPEED GETTER (Synchronisiert mit Atomic Block Standard) ---
static inline const void *vbuf_get_generic(vbuf_instance_t *inst,
                                           uint32_t key_id, size_t *count_out,
                                           uint16_t *width_out) {
  uint8_t *curr = inst->mem + 16;
  uint8_t *end = inst->mem + inst->size;

  while (curr + 16 <= end) {
    uint64_t anchor = *(uint64_t *)curr;
    uint16_t current_id = (uint16_t)((anchor >> 16) & 0xFFFF);
    uint16_t bit_width = (uint16_t)((anchor >> 32) & 0xFFFF);
    uint64_t count = *(uint64_t *)(curr + 8);

    if (anchor == 0) {
      curr += 8;
      continue;
    }

    size_t current_pos = (size_t)(curr - inst->mem);
    size_t align = (size_t)inst->alignment;

    // Diamond Alignment Sprung (Header ist jetzt immer 16 Byte)
    size_t payload_offset = (current_pos + 16 + (align - 1)) & ~(align - 1);

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

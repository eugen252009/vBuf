#include "vbuf.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// --- READING CORE ---

// --- HEADER WRITER ---
void vbuf_write_header(FILE *f, uint8_t a_shift) {
  uint8_t hdr[16] = {0};
  *(uint32_t *)&hdr[0] = MAGIC;
  *(uint32_t *)&hdr[4] = VERSION;
  hdr[8] = a_shift;
  fwrite(hdr, 1, 16, f);
}
// Hilfsfunktion: Sucht den 64-Bit Anchor einer Spalte
static inline const uint8_t *vbuf_find_anchor(vbuf_instance_t *inst,
                                              uint16_t key_id) {
  const uint8_t *curr = inst->mem + 16;
  const uint8_t *end = inst->mem + inst->size;

  while (curr + 16 <=
         end) { // 8 Byte Anchor + mind. 8 Byte N (wegen Overflow bit)
    uint64_t anchor = *(const uint64_t *)curr;
    uint16_t current_id = (uint16_t)((anchor >> 16) & 0xFFFF);
    uint16_t bit_width = (uint16_t)((anchor >> 32) & 0xFFFF);

    // N auslesen (wir nutzen immer das Overflow-Feld direkt nach dem Anchor)
    uint64_t n = *(const uint64_t *)(curr + 8);

    if (anchor == 0) {
      curr += 8;
      continue;
    }

    if (current_id == key_id)
      return curr;

    // Sprung zur nächsten Spalte
    size_t bytes_per_item = bit_width / 8;
    size_t data_start =
        ((size_t)(curr - inst->mem) + 16 + (inst->alignment - 1)) &
        ~(inst->alignment - 1);
    size_t data_bytes = (size_t)n * bytes_per_item;

    size_t next_off = data_start + data_bytes;
    next_off = (next_off + 7) & ~7; // Ausrichtung auf nächsten 64-bit Anchor
    curr = inst->mem + next_off;
  }
  return NULL;
}

vbuf_instance_t *vbuf_open(const char *filename) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0)
    return NULL;

  struct stat st;
  fstat(fd, &st);

  uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  if (map == MAP_FAILED)
    return NULL;

  if (*(uint32_t *)map != 0x46554256) {
    printf("File is not .VBUF (Header: 0x%08X)\n", *(uint32_t *)map);
    munmap(map, st.st_size);
    return NULL;
  }

  vbuf_instance_t *inst = malloc(sizeof(vbuf_instance_t));
  inst->mem = map;
  inst->size = st.st_size;
  inst->alignment = 1 << map[8];
  inst->data = map + 16;

  madvise(map, st.st_size, MADV_HUGEPAGE | MADV_SEQUENTIAL);
  return inst;
}
// --- CLOSE INSTANCE ---
void vbuf_close(vbuf_instance_t *inst) {
  if (!inst)
    return;
  if (inst->mem)
    munmap(inst->mem, inst->size);
  free(inst);
}

// --- WRITING CORE (Multi-Type fähig) ---

void vbuf_write_atomic_column(FILE *f, uint32_t id, size_t n,
                              uint32_t alignment, uint16_t bit_width,
                              const void *data) {
  // 1. Der 64-Bit Anchor (Identität & Typ)
  uint64_t anchor = 0;
  anchor |= (1ULL << 4); // PHYS: Array
  anchor |= (1ULL << 9); // OVERFLOW: N folgt als u64
  anchor |= ((uint64_t)(id & 0xFFFF) << 16);
  anchor |= ((uint64_t)bit_width << 32);

  fwrite(&anchor, 8, 1, f);

  // 2. Die Anzahl (u64)
  uint64_t n_64 = (uint64_t)n;
  fwrite(&n_64, 8, 1, f);

  // 3. Alignment auf das Diamond-Grid
  fflush(f);
  off_t pos = ftello(f);
  size_t pad = (alignment - (pos % alignment)) % alignment;
  if (pad > 0) {
    // Wir nutzen einen statischen Null-Buffer für Speed
    static const uint8_t zero[4096] = {0};
    fwrite(zero, 1, pad, f);
  }

  // 4. Die Daten "dumm" rausschreiben
  size_t bytes_per_item = bit_width / 8;
  fwrite(data, bytes_per_item, n, f);

  // Optionaler Debug-Print
  // printf("[Writer] ID %u: %zu items (%u-bit) written.\n", id, n, bit_width);
}

// --- DISPATCHER ---

const void *vbuf_get_col_ptr(vbuf_instance_t *inst, uint32_t id, size_t *n_out,
                             uint16_t *width_out) {
  const uint8_t *anchor_ptr = vbuf_find_anchor(inst, (uint16_t)id);
  if (!anchor_ptr)
    return NULL;

  uint64_t anchor = *(const uint64_t *)anchor_ptr;
  *n_out = (size_t)(*(const uint64_t *)(anchor_ptr + 8));
  *width_out = (uint16_t)((anchor >> 32) & 0xFFFF);

  size_t data_start =
      ((size_t)(anchor_ptr - inst->mem) + 16 + (inst->alignment - 1)) &
      ~(inst->alignment - 1);
  return (const void *)(inst->mem + data_start);
}
// --- CLASSIC COLUMN WRITER (für Kompatibilität) ---
void vbuf_write_column(FILE *f, uint32_t id, size_t n, uint32_t alignment,
                       const uint16_t *data) {
  // Wir nutzen intern die neue atomare Struktur für Konsistenz
  uint64_t anchor = 0;
  anchor |= (1ULL << 4); // PHYS: Array
  anchor |= (1ULL << 9); // OVERFLOW ACTIVE
  anchor |= ((uint64_t)(id & 0xFFFF) << 16);
  anchor |= (16ULL << 32); // 16 Bit PLen

  fwrite(&anchor, 8, 1, f);
  uint64_t n_64 = (uint64_t)n;
  fwrite(&n_64, 8, 1, f);

  fflush(f);
  off_t pos = ftello(f);
  size_t pad = (alignment - (pos % alignment)) % alignment;
  if (pad > 0) {
    static uint8_t zero_buf[4096] = {0};
    fwrite(zero_buf, 1, pad, f);
  }

  fwrite(data, sizeof(uint16_t), n, f);
}

#include "../vbuf.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int make_path(char *out, size_t size, const char *dir,
                     const char *name) {
  int n = snprintf(out, size, "%s/%s", dir, name);
  return n >= 0 && (size_t)n < size;
}

static const char *const valid_files[] = {
    "valid-basic.vbuf",          "valid-empty.vbuf",
    "valid-zero-array.vbuf",     "valid-duplicate-chain.vbuf",
    "valid-duplicate-unchained.vbuf", "valid-optional-extension.vbuf",
    "valid-indefinite.vbuf",     "valid-writer-primitives.vbuf",
};

static int files_equal(const char *left_path, const char *right_path) {
  FILE *left = fopen(left_path, "rb");
  FILE *right = fopen(right_path, "rb");
  if (!left || !right) {
    if (left) fclose(left);
    if (right) fclose(right);
    return 0;
  }
  int equal = 1;
  for (;;) {
    unsigned char a[4096], b[4096];
    size_t an = fread(a, 1, sizeof(a), left);
    size_t bn = fread(b, 1, sizeof(b), right);
    if (an != bn || memcmp(a, b, an) != 0) { equal = 0; break; }
    if (an < sizeof(a)) break;
  }
  fclose(left);
  fclose(right);
  return equal;
}

static int verify_c_writer(const char *fixture_dir) {
  char output[256], expected[4096];
  snprintf(output, sizeof(output), "/tmp/vbuf-v06-c-writer-%ld.vbuf", (long)getpid());
  if (!make_path(expected, sizeof(expected), fixture_dir, "valid-writer-primitives.vbuf")) return 0;
  uint32_t error = 0;
  vbuf_v06_writer_t *writer = vbuf_v06_writer_create(output, 4, false, &error);
  if (!writer || error != 0) return 0;
#define WRITE(KEY, PHYS, CONT, TYPE, DATA, COUNT) \
  do { if (vbuf_v06_writer_write(writer, KEY, PHYS, CONT, 0, TYPE, DATA, COUNT) != 0) return 0; } while (0)
  const uint8_t u8s[] = {1, 2, 255}; WRITE(10, 1, false, VBUF_V06_U8, u8s, 3);
  const uint16_t u16s[] = {0x1234, 0xabcd}; WRITE(11, 1, false, VBUF_V06_U16, u16s, 2);
  const uint32_t u32s[] = {1, 0xdeadbeef}; WRITE(12, 1, false, VBUF_V06_U32, u32s, 2);
  const uint64_t u64s[] = {1, UINT64_C(0x0102030405060708)}; WRITE(13, 1, false, VBUF_V06_U64, u64s, 2);
  const int8_t i8s[] = {-1, 2}; WRITE(14, 1, false, VBUF_V06_I8, i8s, 2);
  const int16_t i16s[] = {-2, 3}; WRITE(15, 1, false, VBUF_V06_I16, i16s, 2);
  const int32_t i32s[] = {-3, 4}; WRITE(16, 1, false, VBUF_V06_I32, i32s, 2);
  const int64_t i64s[] = {-4, 5}; WRITE(17, 1, false, VBUF_V06_I64, i64s, 2);
  const float f32s[] = {1.5f, -2.25f}; WRITE(18, 1, false, VBUF_V06_F32, f32s, 2);
  const double f64s[] = {3.5, -4.75}; WRITE(19, 1, false, VBUF_V06_F64, f64s, 2);
  const uint8_t opaque[] = {'a', 'b', 'c'}; WRITE(20, 1, false, VBUF_V06_OPAQUE, opaque, 3);
  const uint32_t scalar[] = {42}; WRITE(21, 0, false, VBUF_V06_U32, scalar, 1);
  const uint8_t a[] = {'A'}, b[] = {'B'}; WRITE(22, 1, true, VBUF_V06_OPAQUE, a, 1); WRITE(22, 1, false, VBUF_V06_OPAQUE, b, 1);
#undef WRITE
  if (vbuf_v06_writer_finish(writer) != 0) return 0;
  int equal = files_equal(output, expected);
  remove(output);
  if (!equal) return 0;

  writer = vbuf_v06_writer_create(output, 3, true, &error);
  if (!writer || vbuf_v06_writer_write(writer, 1, 1, false, 0,
                                        VBUF_V06_U8, u8s, 3) != 0 ||
      vbuf_v06_writer_finish(writer) != 0)
    return 0;
  vbuf_v06_instance_t *instance = vbuf_v06_open(output, &error);
  if (!instance) return 0;
  vbuf_v06_close(instance);
  remove(output);

  writer = vbuf_v06_writer_create(output, 3, false, &error);
  if (!writer || vbuf_v06_writer_write(writer, 7, 1, true, 0,
                                        VBUF_V06_U8, a, 1) != 0 ||
      vbuf_v06_writer_write(writer, 8, 1, false, 0,
                             VBUF_V06_U8, b, 1) == 0 ||
      vbuf_v06_writer_write(writer, 7, 1, false, 0,
                             VBUF_V06_U8, b, 1) != 0 ||
      vbuf_v06_writer_finish(writer) != 0)
    return 0;
  remove(output);

  writer = vbuf_v06_writer_create(output, 3, false, &error);
  if (!writer || vbuf_v06_writer_write(writer, 7, 1, true, 0,
                                        VBUF_V06_U8, a, 1) != 0 ||
      vbuf_v06_writer_finish(writer) == 0)
    return 0;
  remove(output);
  return 1;
}

static const char *const invalid_files[] = {
    "bad-magic.vbuf",
    "base-shift-high.vbuf",
    "base-shift-low.vbuf",
    "continuation-final.vbuf",
    "chain-key-mismatch.vbuf",
    "count-overflow.vbuf",
    "final-tail-padding.vbuf",
    "impossible-scalar-count.vbuf",
    "indefinite-nonzero-size.vbuf",
    "indefinite-truncated.vbuf",
    "invalid-header-size.vbuf",
    "invalid-payload-shift.vbuf",
    "invalid-physical.vbuf",
    "invalid-semantic.vbuf",
    "known-size-mismatch.vbuf",
    "known-size-over-4g.vbuf",
    "known-size-overflow.vbuf",
    "malformed-extension-overrun.vbuf",
    "malformed-extension-zero.vbuf",
    "noncanonical-extended-inline.vbuf",
    "noncanonical-extended-small.vbuf",
    "nonzero-global-padding.vbuf",
    "nonzero-payload-padding.vbuf",
    "payload-truncated-byte.vbuf",
    "payload-truncated-element.vbuf",
    "reserved-nonzero.vbuf",
    "short-header.vbuf",
    "truncated-anchor.vbuf",
    "truncated-extended-count.vbuf",
    "unaligned-block-start.vbuf",
    "unknown-flags.vbuf",
    "unknown-required-extension.vbuf",
    "unsupported-version.vbuf",
    "unsupported-width.vbuf",
};

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s FIXTURE_DIR\n", argv[0]);
    return 2;
  }

  char path[4096];
  for (size_t i = 0; i < sizeof(valid_files) / sizeof(valid_files[0]); ++i) {
    if (!make_path(path, sizeof(path), argv[1], valid_files[i]))
      return 1;
    uint32_t error = UINT32_MAX;
    vbuf_v06_instance_t *instance = vbuf_v06_open(path, &error);
    if (!instance || error != 0) {
      fprintf(stderr, "valid fixture rejected: %s (error %u)\n", valid_files[i],
              error);
      return 1;
    }
    vbuf_v06_close(instance);
  }

  for (size_t i = 0; i < sizeof(invalid_files) / sizeof(invalid_files[0]); ++i) {
    if (!make_path(path, sizeof(path), argv[1], invalid_files[i]))
      return 1;
    uint32_t error = 0;
    vbuf_v06_instance_t *instance = vbuf_v06_open(path, &error);
    if (instance || error == 0) {
      fprintf(stderr, "invalid fixture accepted: %s\n", invalid_files[i]);
      if (instance)
        vbuf_v06_close(instance);
      return 1;
    }
  }

  if (!make_path(path, sizeof(path), argv[1], "valid-basic.vbuf"))
    return 1;
  uint32_t error = 0;
  vbuf_v06_instance_t *instance = vbuf_v06_open(path, &error);
  if (!instance)
    return 1;
  uint64_t base_step = 0, data_start = 0, data_size = 0;
  if (vbuf_v06_header_info(instance, &base_step, &data_start, &data_size) != 0 ||
      base_step != 8 || data_start != 24 || data_size != 20) {
    vbuf_v06_close(instance);
    return 1;
  }
  uint64_t block_start = 0, payload_start = 0, payload_length = 0;
  uint64_t next_block_start = 0, wire_count = 0;
  if (vbuf_v06_block_info(instance, 0, &block_start, &payload_start,
                          &payload_length, &next_block_start, &wire_count) != 0 ||
      block_start != 24 || payload_start != 32 || payload_length != 12 ||
      next_block_start != 48 || wire_count != 3) {
    vbuf_v06_close(instance);
    return 1;
  }

  const void *payload = (const void *)(uintptr_t)1;
  size_t count = SIZE_MAX;
  error = vbuf_v06_get(instance, 1, 0, VBUF_V06_U32, &payload, &count);
  const uint32_t *values = (const uint32_t *)payload;
  if (error != 0 || !values || count != 3 || values[0] != 1 ||
      values[1] != 2 || values[2] != 3) {
    vbuf_v06_close(instance);
    return 1;
  }

  payload = (const void *)(uintptr_t)1;
  count = SIZE_MAX;
  error = vbuf_v06_get(instance, 1, 0, VBUF_V06_F32, &payload, &count);
  if (error == 0 || payload != NULL || count != 0) {
    vbuf_v06_close(instance);
    return 1;
  }
  vbuf_v06_close(instance);

  if (!make_path(path, sizeof(path), argv[1], "valid-zero-array.vbuf"))
    return 1;
  instance = vbuf_v06_open(path, &error);
  payload = NULL;
  count = SIZE_MAX;
  if (!instance || vbuf_v06_get(instance, 2, 0, VBUF_V06_U8, &payload,
                                &count) != 0 ||
      payload == NULL || count != 0) {
    if (instance)
      vbuf_v06_close(instance);
    return 1;
  }
  vbuf_v06_close(instance);

  if (!verify_c_writer(argv[1]))
    return 1;

  puts("validated 42 shared vBuf-v0.6 outcomes, complete C pointer ranges, and canonical C writer bytes");
  return 0;
}

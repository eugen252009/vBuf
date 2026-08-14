#include "../vbuf.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int make_path(char *out, size_t size, const char *dir,
                     const char *name) {
  int n = snprintf(out, size, "%s/%s", dir, name);
  return n >= 0 && (size_t)n < size;
}

static const char *const valid_files[] = {
    "valid-basic.vbuf",          "valid-empty.vbuf",
    "valid-zero-array.vbuf",     "valid-duplicate-chain.vbuf",
    "valid-duplicate-unchained.vbuf", "valid-optional-extension.vbuf",
    "valid-indefinite.vbuf",
};

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

  puts("validated 41 shared vBuf-v0.6 outcomes and complete C pointer ranges");
  return 0;
}

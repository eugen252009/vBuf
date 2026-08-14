#include "../vbuf.h"

#include <stdint.h>
#include <stdio.h>

static int make_path(char *out, size_t size, const char *dir,
                     const char *name) {
  int n = snprintf(out, size, "%s/%s", dir, name);
  return n >= 0 && (size_t)n < size;
}

static int verify_rust_fixture(const char *path) {
  vbuf_instance_t *inst = vbuf_open(path);
  if (!inst)
    return 0;
  size_t count = 0;
  uint16_t width = 0;
  const uint32_t *a =
      (const uint32_t *)vbuf_get_generic(inst, 1, &count, &width);
  int ok = a && count == 3 && width == 32 && a[0] == 1 && a[1] == 2 &&
           a[2] == 3;
  const uint16_t *b =
      (const uint16_t *)vbuf_get_generic(inst, 2, &count, &width);
  ok = ok && b && count == 2 && width == 16 && b[0] == 500 && b[1] == 1000;
  vbuf_close(inst);
  return ok;
}

static int verify_typescript_fixture(const char *path) {
  vbuf_instance_t *inst = vbuf_open(path);
  if (!inst)
    return 0;
  size_t count = 0;
  uint16_t width = 0;
  const int32_t *a =
      (const int32_t *)vbuf_get_generic(inst, 1, &count, &width);
  int ok = a && count == 3 && width == 32 && a[0] == 1 && a[1] == 2 &&
           a[2] == 3;
  const double *b =
      (const double *)vbuf_get_generic(inst, 2, &count, &width);
  ok = ok && b && count == 2 && width == 64 && b[0] == 1.5 && b[1] == -2.25;
  vbuf_close(inst);
  return ok;
}

static int observe_truncated_pointer(const char *path) {
  vbuf_instance_t *inst = vbuf_open(path);
  if (!inst)
    return 0;
  size_t count = 0;
  uint16_t width = 0;
  const void *ptr = vbuf_get_generic(inst, 2, &count, &width);

  // Legacy C-facing behavior exposes a pointer/count for a 16-byte declared
  // range with only 8 bytes available. Never dereference it as two doubles.
  // Step 4 must require v0.6 rejection before returning this pointer.
  int observed = ptr != NULL && count == 2 && width == 64;
  vbuf_close(inst);
  return observed;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s FIXTURE_DIR\n", argv[0]);
    return 2;
  }

  char path[4096];
  if (!make_path(path, sizeof(path), argv[1], "current-rust.vbuf") ||
      !verify_rust_fixture(path))
    return 1;
  if (!make_path(path, sizeof(path), argv[1], "current-typescript.vbuf") ||
      !verify_typescript_fixture(path))
    return 1;
  if (!make_path(path, sizeof(path), argv[1], "short.vbuf"))
    return 1;
  vbuf_instance_t *short_file = vbuf_open(path);
  if (short_file) {
    vbuf_close(short_file);
    return 1;
  }
  if (!make_path(path, sizeof(path), argv[1], "bad-magic.vbuf"))
    return 1;
  vbuf_instance_t *bad_magic = vbuf_open(path);
  if (bad_magic) {
    vbuf_close(bad_magic);
    return 1;
  }
  if (!make_path(path, sizeof(path), argv[1],
                 "truncated-typescript.vbuf") ||
      !observe_truncated_pointer(path))
    return 1;

  puts("validated Rust/C/TypeScript legacy-v0.5 byte and behavior evidence");
  return 0;
}

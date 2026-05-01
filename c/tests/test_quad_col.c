#include "test_interface.h"
#include <stdio.h>
#include <time.h>

void run_test(vbuf_instance_t *inst) {
  size_t n[4];
  const uint16_t *cols[4];
  for (int i = 0; i < 4; i++) {
    cols[i] = vbuf_get_u16(inst, 101 + i, &n[i]);
  }

  if (!cols[0] || !cols[1] || !cols[2] || !cols[3]) {
    printf("[Plugin] Fehler: Nicht alle 4 Spalten (101-104) gefunden!\n");
    return;
  }

  double total = 0;
  struct timespec start_t, end_t;

  printf("\n--- vBuf Quad-Column (A+B)*(C-D) Benchmark ---\n");

  clock_gettime(CLOCK_MONOTONIC, &start_t);

#pragma omp parallel for reduction(+ : total) schedule(static, 1024 * 1024)
  for (size_t i = 0; i < n[0]; i++) {
    total +=
        (double)(cols[0][i] + cols[1][i]) * (double)(cols[2][i] - cols[3][i]);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_t);

  double time =
      (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_nsec - start_t.tv_nsec) / 1e9;
  double bytes = (double)n[0] * sizeof(uint16_t) * 4; // 4 Spalten

  printf("Speed:    \033[1;31m%.2f GB/s\033[0m\n", (bytes / time) / 1e9);
  printf("Zeit:     %.4f s\n", time);
}

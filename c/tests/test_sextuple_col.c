#include "test_interface.h"
#include <stdio.h>
#include <time.h>

void run_test(vbuf_instance_t *inst) {
  size_t n_elements = 0;
  const uint16_t *cols[6];

  // 1. Alle 6 Spalten laden (101 bis 106)
  // Wenn du IDs 101 bis 106 geschrieben hast:
  for (int i = 0; i < 6; i++) {
    cols[i] = vbuf_get_u16(inst, 101 + i, &n_elements);
    if (!cols[i])
      printf("Fehlgeschlagen bei ID %d\n", 101 + i);
  }

  // 2. Alle Spalten validieren
  if (!cols[0] || !cols[1] || !cols[2] || !cols[3] || !cols[4] || !cols[5]) {
    printf("[Plugin] Fehler: Nicht alle 6 Spalten (101-106) gefunden!\n");
    return;
  }

  double total = 0;
  struct timespec start_t, end_t;

  printf("\n--- vBuf Sextuple-Column Benchmark (6 Spalten) ---\n");

  clock_gettime(CLOCK_MONOTONIC, &start_t);

// 3. Korrektes Pragma und Schleife
#pragma omp parallel for reduction(+ : total) schedule(static, 1024 * 1024)
  for (size_t i = 0; i < n_elements; i++) {
    double part1 = (double)cols[0][i] + (double)cols[1][i];
    double part2 = (double)cols[2][i] - (double)cols[3][i];
    double part3 = (double)cols[4][i] + (double)cols[5][i];

    total += (part1 * part2) + part3;
  }

  clock_gettime(CLOCK_MONOTONIC, &end_t);

  double time =
      (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_nsec - start_t.tv_nsec) / 1e9;

  // 4. Bandbreite berechnen: N * 2 Byte * 6 Spalten
  double bytes = (double)n_elements * sizeof(uint16_t) * 6;

  printf("Speed:    \033[1;32m%.2f GB/s\033[0m\n", (bytes / time) / 1e9);
  printf("Zeit:     %.4f s\n", time);
  printf("Check:    Total sum is %.2f\n", total); // Verhindert Wegoptimierung
}

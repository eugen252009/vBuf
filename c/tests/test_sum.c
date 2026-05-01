#include "test_interface.h"
#include <stdio.h>
#include <time.h>

void run_test(vbuf_instance_t *inst) {
  size_t n;
  const uint16_t *data = vbuf_get_u16(inst, 101, &n);
  if (!data)
    return;

  uint64_t total_sum = 0;
  struct timespec start_t, end_t;

  printf("\n--- vBuf Integer Sum Benchmark ---\n");

  clock_gettime(CLOCK_MONOTONIC, &start_t);

#pragma omp parallel for reduction(+ : total_sum) schedule(static, 1024 * 1024)
  for (size_t i = 0; i < n; i += 4) {
    total_sum += data[i];
    total_sum += data[i + 1];
    total_sum += data[i + 2];
    total_sum += data[i + 3];
  }

  clock_gettime(CLOCK_MONOTONIC, &end_t);

  double time =
      (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_nsec - start_t.tv_nsec) / 1e9;
  printf("Summe:   %lu\n", total_sum);
  printf("Speed:   \033[1;35m%.2f GB/s\033[0m\n",
         (n * sizeof(uint16_t) / time) / 1e9);
  printf("Zeit:    %.4f s\n", time);
}

#include "test_interface.h"
#include <stdio.h>
#include <time.h>

void run_test(vbuf_instance_t *inst) {
    size_t n1, n2, n3;
    const uint16_t *a = vbuf_get_u16(inst, 101, &n1);
    const uint16_t *b = vbuf_get_u16(inst, 102, &n2);
    const uint16_t *c = vbuf_get_u16(inst, 103, &n3);

    if (!a || !b || !c) {
        printf("[Plugin] Fehler: Spalte 101, 102 oder 103 fehlt!\n");
        return;
    }

    double total = 0;
    struct timespec start_t, end_t;

    printf("\n--- vBuf Triple-Column (A * B + C) Benchmark ---\n");

    clock_gettime(CLOCK_MONOTONIC, &start_t);

    #pragma omp parallel for reduction(+:total) schedule(static, 1024*1024)
    for (size_t i = 0; i < n1; i++) {
        total += ((double)a[i] * (double)b[i]) + (double)c[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end_t);

    double time = (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_nsec - start_t.tv_nsec) / 1e9;
    double bytes = (double)n1 * sizeof(uint16_t) * 3; // 3 Spalten

    printf("Speed:    \033[1;33m%.2f GB/s\033[0m\n", (bytes / time) / 1e9);
    printf("Zeit:     %.4f s\n", time);
}

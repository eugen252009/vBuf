#include "test_interface.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <x86intrin.h>

void benchmark_std_dev(const uint16_t *data, size_t n) {
    size_t distances[] = {0, 256, 512, 1024, 2048, 4096};
    unsigned int d;

    printf("\n--- vBuf Statistik Performance Test (Prefetch-Check) ---\n");
    printf("Werte: %zu | Kern-Check...\n", n);

    for (int d_idx = 0; d_idx < 6; d_idx++) {
        size_t dist = distances[d_idx];
        double sum = 0, sum_sq = 0;
        uint64_t start_c, end_c;
        struct timespec start_t, end_t;

        printf("Test mit Distanz: %zu\n", dist);

        clock_gettime(CLOCK_MONOTONIC, &start_t);
        start_c = __rdtscp(&d);

        #pragma omp parallel for reduction(+ : sum, sum_sq) schedule(static, 1024 * 1024)
        for (size_t i = 0; i < n; i++) {
            // Software Prefetching
            if (dist > 0 && (i & 31) == 0) {
                __builtin_prefetch(&data[i + dist], 0, 3);
            }

            double val = (double)data[i];
            sum += val;
            sum_sq += val * val;
        }

        end_c = __rdtscp(&d);
        clock_gettime(CLOCK_MONOTONIC, &end_t);

        double mean = sum / n;
        double variance = (sum_sq / n) - (mean * mean);
        double std_dev = sqrt(variance);
        double time = (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_nsec - start_t.tv_nsec) / 1e9;

        printf("Ergebnis:    %.4f\n", std_dev);
        printf("Speed:       \033[1;32m%.2f GB/s\033[0m\n", (n * sizeof(uint16_t) / time) / 1e9);
        printf("Cycles/Itm:  %.3f\n", (double)(end_c - start_c) / n);
        printf("----------------------------------------\n");
    }
}

// Das Plugin-Interface für den vbuf_test_runner
void run_test(vbuf_instance_t *inst) {
    uint32_t id = 101;
    size_t n;
    
    const uint16_t *data = vbuf_get_u16(inst, id, &n);

    if (!data) {
        printf("[Plugin] Fehler: Spalte %u nicht gefunden.\n", id);
        return;
    }

    benchmark_std_dev(data, n);
}

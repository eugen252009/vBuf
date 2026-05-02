#include "test_interface.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

void run_test(vbuf_instance_t *inst) {
    size_t n;
    uint16_t width;
    // Wir nutzen den generischen Pointer und casten auf u32
    const uint32_t *data = (const uint32_t *)vbuf_get_generic(inst, 101, &n, &width);
    
    if (!data || width != 32) {
        printf("Fehler: Spalte 101 nicht gefunden oder keine 32-bit Breite (Breite: %u)\n", width);
        return;
    }

    uint64_t total_sum = 0;
    struct timespec start_t, end_t;

    printf("\n--- vBuf Integer Sum Benchmark (C / OpenMP 32-bit) ---\n");

    clock_gettime(CLOCK_MONOTONIC, &start_t);

    // OpenMP Parallelisierung mit Static Scheduling für maximale Cache-Effizienz
    #pragma omp parallel for reduction(+ : total_sum) schedule(static, 1024 * 1024)
    for (size_t i = 0; i < n; i++) {
        total_sum += data[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end_t);

    double time = (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_nsec - start_t.tv_nsec) / 1e9;
    
    // Datengröße in GB (200 Mio * 4 Byte)
    double bytes_processed = (double)n * sizeof(uint32_t);
    double speed_gb_s = (bytes_processed / time) / 1e9;

    printf("Summe:   %lu\n", total_sum);
    printf("Speed:   \033[1;35m%.2f GB/s\033[0m\n", speed_gb_s);
    printf("Zeit:    %.4f s\n", time);
}

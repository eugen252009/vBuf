#include "test_interface.h" // Wichtig: Hier ist vbuf_test_func definiert
#include <stdio.h>
#include <time.h>

void benchmark_filter_performance(const uint16_t *a, const uint16_t *b, size_t n) {
    size_t match_count = 0;
    struct timespec start_t, end_t;

    printf("\n--- vBuf Branch-Killer Benchmark ---\n");
    printf("Vergleiche A (Seq) vs B (Random): if(A[i] > B[i]) count++\n");

    clock_gettime(CLOCK_MONOTONIC, &start_t);

    #pragma omp parallel for reduction(+:match_count) schedule(static, 1024*1024)
    for (size_t i = 0; i < n; i++) {
        // Branch-Prediction Test
        if (a[i] > b[i]) {
            match_count++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_t);

    double time = (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_nsec - start_t.tv_nsec) / 1e9;
    double bytes_processed = (double)n * sizeof(uint16_t) * 2;

    printf("Matches:    %zu\n", match_count);
    printf("Speed:      \033[1;33m%.2f GB/s\033[0m\n", (bytes_processed / time) / 1e9);
    printf("Zeit:       %.4f s\n", time);
    printf("------------------------------------\n");
}

// DAS HIER IST DER EINSPRUNGSPUNKT FÜR DEN RUNNER
void run_test(vbuf_instance_t *inst) {
    size_t n1, n2;
    const uint16_t *col_a = vbuf_get_u16(inst, 101, &n1);
    const uint16_t *col_b = vbuf_get_u16(inst, 102, &n2);

    if (col_a && col_b) {
        benchmark_filter_performance(col_a, col_b, n1);
    } else {
        printf("[Plugin] Fehler: Konnte Spalten 101 oder 102 nicht finden!\n");
    }
}

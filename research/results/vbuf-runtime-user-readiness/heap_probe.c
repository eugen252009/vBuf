// Linux/glibc diagnostic preload, not linked into the production runtime.
#define _GNU_SOURCE
#include <malloc.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static pthread_t worker;
static atomic_int stopped;
static int started;

static void * sample(void * unused) {
    (void)unused;
    FILE * log = fopen(getenv("VBUF_HEAP_PROBE_LOG"), "w");
    if (!log) return NULL;
    while (!atomic_load(&stopped)) {
        const struct mallinfo2 info = mallinfo2();
        struct timespec time;
        clock_gettime(CLOCK_MONOTONIC, &time);
        fprintf(log, "{\"monotonic_seconds\":%.6f,\"arena_bytes\":%zu,\"arena_used_bytes\":%zu,"
            "\"arena_free_bytes\":%zu,\"malloc_mmap_bytes\":%zu}\n",
            time.tv_sec + time.tv_nsec / 1e9, info.arena, info.uordblks, info.fordblks, info.hblkhd);
        fflush(log);
        if (getenv("VBUF_HEAP_TRIM_CONTROL")) malloc_trim(0);
        const struct timespec delay = {0, 500000000};
        nanosleep(&delay, NULL);
    }
    fclose(log);
    return NULL;
}

__attribute__((constructor)) static void start(void) {
    if (getenv("VBUF_HEAP_PROBE_LOG")) started = pthread_create(&worker, NULL, sample, NULL) == 0;
}

__attribute__((destructor)) static void finish(void) {
    if (started) { atomic_store(&stopped, 1); pthread_join(worker, NULL); }
}

#include "vbuf_region_executor.h"

#include <cstdio>

#include "ggml-backend.h"

extern "C" const char * vbuf_region_executor_backend_name(void) {
    static char name[64];
    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        return "unavailable";
    }
    std::snprintf(name, sizeof(name), "%s", ggml_backend_name(backend));
    ggml_backend_free(backend);
    return name;
}

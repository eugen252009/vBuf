#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Substrate-only capability query. Region lowering and execution semantics are
// intentionally not part of this initial build boundary.
const char * vbuf_region_executor_backend_name(void);

#ifdef __cplusplus
}
#endif

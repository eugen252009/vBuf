#pragma once

#include <cstdint>
#include <cstddef>

extern "C" {

enum {
    VBUF_PORTABLE_EXEC_ABI_V1 = 1,
    VBUF_PORTABLE_EXEC_ABI_V2 = 2,
    VBUF_FFI_OK = 0,
    VBUF_FFI_INVALID_ARGUMENT = 1,
    VBUF_FFI_INVALID_UTF8 = 2,
    VBUF_FFI_INVALID_GRAPH = 3,
    VBUF_FFI_UNSUPPORTED = 4,
    VBUF_FFI_MISSING_BINDING = 5,
    VBUF_FFI_INVALID_ATTRIBUTES = 6,
    VBUF_FFI_MISSING_STATE = 7,
    VBUF_ACTIVATION_SILU = 1,
};

struct VbufRuntimeGraphHandle;

struct VbufFfiBytes { const uint8_t * ptr; uint64_t len; };
static_assert(sizeof(VbufFfiBytes) == 16, "portable graph ABI bytes layout changed");

struct VbufPortableBindingDesc {
    VbufFfiBytes semantic;
    uint32_t tensor_id;
};

struct VbufPortableInputDesc {
    uint8_t kind;
    uint8_t reserved[3];
    uint32_t id;
    VbufFfiBytes semantic;
};

struct VbufPortableOperationDesc {
    uint32_t kind;
    VbufFfiBytes id;
    const VbufPortableInputDesc * inputs;
    uint32_t input_count;
    uint32_t output;
    uint8_t has_epsilon;
    uint8_t matmul_weight_operand;
    uint8_t matmul_transpose_weight;
    uint8_t top_k_order;
    uint8_t top_k_tie_break;
    // For kind 3, reserved[0] is VBUF_ACTIVATION_SILU.
    uint8_t reserved[3];
    float epsilon;
    uint8_t has_top_k;
    uint8_t reserved_top_k[3];
    uint32_t top_k;
};

struct VbufPortableProgramDesc {
    const VbufPortableBindingDesc * bindings;
    uint32_t binding_count;
};

struct VbufPortableRegionDesc {
    uint32_t input;
    uint32_t output;
    const VbufPortableOperationDesc * operations;
    uint32_t operation_count;
};

struct VbufAttentionDesc {
    uint64_t batch_size;
    uint64_t query_head_count;
    uint64_t kv_head_count;
    uint64_t head_dim;
    uint64_t query_length;
    uint64_t current_kv_length;
    float scale;
    // 0 = none, 1 = causal.
    uint8_t mask_kind;
    // 1 = state length.
    uint8_t position_kind;
    uint8_t reserved[2];
    uint32_t state_id;
};

struct VbufPortableOperationDescV2 {
    VbufPortableOperationDesc base;
    VbufAttentionDesc attention;
};

struct VbufPortableRegionDescV2 {
    uint32_t input;
    uint32_t output;
    const VbufPortableOperationDescV2 * operations;
    uint32_t operation_count;
};

struct VbufFfiError { uint32_t code; VbufFfiBytes message; };

struct VbufGraphTensorDesc { uint32_t tensor_id; VbufFfiBytes semantic; };

struct VbufGraphInputDesc {
    uint8_t kind;
    uint8_t reserved[3];
    uint32_t id;
};

struct VbufGraphOperationDesc {
    // 1 = RmsNorm, 2 = MatMul, 3 = Activation, 4 = TopK,
    // 5 = IndexedMatMul, 6 = Attention, 7 = ResidualAdd.
    uint32_t kind;
    VbufFfiBytes id;
    const VbufGraphInputDesc * inputs;
    uint32_t input_count;
    uint32_t output;
    uint8_t has_epsilon;
    uint8_t matmul_weight_operand;
    uint8_t matmul_transpose_weight;
    uint8_t top_k_order;
    uint8_t top_k_tie_break;
    float epsilon;
    uint8_t has_top_k;
    // For kind 3, reserved[0] is VBUF_ACTIVATION_SILU.
    uint8_t reserved[3];
    uint32_t top_k;
};

struct VbufGraphOperationDescV2 {
    VbufGraphOperationDesc base;
    VbufAttentionDesc attention;
};

static_assert(sizeof(VbufPortableOperationDesc) == 64, "portable operation ABI layout changed");
static_assert(sizeof(VbufPortableOperationDescV2) == 128, "portable operation v2 ABI layout changed");
static_assert(sizeof(VbufGraphOperationDesc) == 64, "graph operation ABI layout changed");
static_assert(sizeof(VbufGraphOperationDescV2) == 128, "graph operation v2 ABI layout changed");
static_assert(sizeof(VbufPortableInputDesc) == 24, "portable input ABI layout changed");
static_assert(sizeof(VbufAttentionDesc) == 64, "attention ABI layout changed");

uint32_t vbuf_runtime_graph_abi_version();
uint32_t vbuf_runtime_graph_lower_v1(const VbufPortableProgramDesc *,
    const VbufPortableRegionDesc *, VbufRuntimeGraphHandle **, VbufFfiError *);
uint32_t vbuf_runtime_graph_lower_v2(const VbufPortableProgramDesc *,
    const VbufPortableRegionDescV2 *, VbufRuntimeGraphHandle **, VbufFfiError *);
void vbuf_runtime_graph_close(VbufRuntimeGraphHandle *);
uint32_t vbuf_runtime_graph_operation_count(const VbufRuntimeGraphHandle *);
uint32_t vbuf_runtime_graph_tensor_count(const VbufRuntimeGraphHandle *);
uint32_t vbuf_runtime_graph_input_value(const VbufRuntimeGraphHandle *, uint32_t *);
uint32_t vbuf_runtime_graph_output_value(const VbufRuntimeGraphHandle *, uint32_t *);
uint32_t vbuf_runtime_graph_tensor_desc(const VbufRuntimeGraphHandle *, uint32_t,
    VbufGraphTensorDesc *);
uint32_t vbuf_runtime_graph_operation_desc(const VbufRuntimeGraphHandle *, uint32_t,
    VbufGraphOperationDesc *);
uint32_t vbuf_runtime_graph_operation_desc_v2(const VbufRuntimeGraphHandle *, uint32_t,
    VbufGraphOperationDescV2 *);

}

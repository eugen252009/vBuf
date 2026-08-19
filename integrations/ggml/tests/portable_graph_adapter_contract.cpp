#include "vbuf_portable_graph_adapter.h"

#include <cassert>
#include <memory>

struct VbufRuntimeGraphHandle {};

namespace {

VbufGraphInputDesc topk_input{ 0, { 0, 0, 0 }, 0 };
bool invalid_topk = false;

} // namespace

extern "C" uint32_t vbuf_runtime_graph_abi_version() { return VBUF_PORTABLE_EXEC_ABI_V1; }
extern "C" uint32_t vbuf_runtime_graph_input_value(const VbufRuntimeGraphHandle *, uint32_t * value) {
    *value = 0;
    return VBUF_FFI_OK;
}
extern "C" uint32_t vbuf_runtime_graph_output_value(const VbufRuntimeGraphHandle *, uint32_t * value) {
    *value = 3;
    return VBUF_FFI_OK;
}
extern "C" uint32_t vbuf_runtime_graph_operation_count(const VbufRuntimeGraphHandle *) { return 1; }
extern "C" uint32_t vbuf_runtime_graph_operation_desc(const VbufRuntimeGraphHandle *, uint32_t,
    VbufGraphOperationDesc * output) {
    *output = {};
    output->kind = 4;
    output->inputs = &topk_input;
    output->input_count = 1;
    output->output = 3;
    output->top_k_order = invalid_topk ? 0 : 1;
    output->top_k_tie_break = 1;
    output->has_top_k = 1;
    output->top_k = 2;
    return VBUF_FFI_OK;
}

int main() {
    VbufRuntimeGraphHandle graph;
    const vbuf_ggml::PortableActivation input{ { 0.25f, 3.0f, 3.0f, -1.0f }, { 4, 1 } };
    const std::shared_ptr<const void> lease = std::make_shared<int>(1);
    const vbuf_ggml::PortableTensorResolver resolver = [](uint32_t, vbuf_ggml::PortableResolvedTensor *, std::string *) {
        return false;
    };
    vbuf_ggml::PortableRouterPrefixResult result;
    std::string error;
    assert(vbuf_ggml::execute_portable_graph(&graph, input, lease, resolver, &result, &error));
    assert((result.selection.ids == std::vector<uint32_t>{ 1, 2 }));
    assert(result.selection.scores.size() == 2);

    invalid_topk = true;
    result = {};
    error.clear();
    assert(!vbuf_ggml::execute_portable_graph(&graph, input, lease, resolver, &result, &error));
    assert(error == "portable TopK semantics are incomplete");
    return 0;
}

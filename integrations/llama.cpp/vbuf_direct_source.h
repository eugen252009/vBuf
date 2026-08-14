#pragma once

#include "llama-model-source.h"

#include <memory>

namespace vbuf_llama {

struct Step28LayerPreparation {
    uint64_t layer_id = 0;
    uint64_t start_offset = 0;
    uint64_t end_offset = 0;
    uint64_t span_bytes = 0;
    uint64_t useful_bytes = 0;
    uint64_t gap_bytes = 0;
    uint64_t tensor_count = 0;
    double prepare_ms = 0;
    long minor_faults = 0;
    long major_faults = 0;
};

struct Step28PhysicalSpan {
    std::string role;
    uint64_t layer_id = UINT64_MAX;
    uint64_t start_offset = 0;
    uint64_t end_offset = 0;
    uint64_t span_bytes = 0;
    uint64_t useful_bytes = 0;
    uint64_t gap_bytes = 0;
    uint64_t tensor_count = 0;
};

struct Step28PreparationResult {
    uint64_t layer_count = 0;
    uint64_t span_count = 0;
    uint64_t global_span_count = 0;
    uint64_t covered_bytes = 0;
    uint64_t useful_bytes = 0;
    uint64_t gap_bytes = 0;
    uint64_t pages_touched = 0;
    uint64_t touch_operations = 0;
    uint64_t prepared_span_bytes = 0;
    uint64_t prepared_useful_bytes = 0;
    uint64_t page_size = 0;
    double prepare_ms = 0;
    long minor_faults = 0;
    long major_faults = 0;
    std::vector<Step28LayerPreparation> layers;
    std::vector<Step28PhysicalSpan> spans;
};

std::shared_ptr<llama_model_source> make_vbuf_direct_source(const char * path);
void set_vbuf_direct_tensor_data(ggml_tensor * tensor, void * userdata);
bool step28_prepare_vbuf_source(llama_model_source * source, const char * variant, Step28PreparationResult & result);

} // namespace vbuf_llama

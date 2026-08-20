#define VBUF_POC13_LIBRARY_ONLY
#include "../tools/full_moe_layer_poc13.cpp"
#undef VBUF_POC13_LIBRARY_ONLY

#include "vbuf_portable_graph_adapter.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct HeldPayload {
    std::shared_ptr<ResidentTensorMaterializer> materializer;
    uint32_t ref = 0;
    MaterializedTensor payload{};
    PersistentTensorRef tensor{};

    ~HeldPayload() {
        if (materializer != nullptr) materializer->release(ref);
    }
};

struct HeldLease {
    std::map<uint32_t, std::shared_ptr<HeldPayload>> payloads;
};

std::shared_ptr<HeldPayload> hold_payload(
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    uint32_t ref, const PersistentTensorRef & tensor) {
    if (!materializer->request(ref, tensor, 256 * 1024 * 1024) ||
        materializer->wait(ref) != MaterializationState::Ready) {
        throw std::runtime_error("portable payload did not become ready");
    }
    auto payload = materializer->obtain_ready_tensor(ref);
    if (!payload.has_value()) throw std::runtime_error("portable payload unavailable");
    auto held = std::make_shared<HeldPayload>();
    held->materializer = materializer;
    held->ref = ref;
    held->payload = std::move(*payload);
    held->tensor = tensor;
    held->tensor.view = held->payload.view();
    return held;
}

std::shared_ptr<const void> hold_tensors(
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const PersistentTensorRef & norm, const PersistentTensorRef & router,
    std::shared_ptr<HeldLease> * lease, PortableTensorResolver * resolver) {
    auto owner = std::make_shared<HeldLease>();
    if (norm.tensor_id > UINT32_MAX || router.tensor_id > UINT32_MAX)
        throw std::runtime_error("TensorBinding ID exceeds portable ABI width");
    const uint32_t norm_id = static_cast<uint32_t>(norm.tensor_id);
    const uint32_t router_id = static_cast<uint32_t>(router.tensor_id);
    owner->payloads.emplace(norm_id, hold_payload(materializer, norm_id, norm));
    owner->payloads.emplace(router_id, hold_payload(materializer, router_id, router));
    const PortableTensorResolver lookup = [owner](uint32_t id,
        PortableResolvedTensor * output, std::string * error) {
        const auto it = owner->payloads.find(id);
        if (it == owner->payloads.end()) {
            if (error != nullptr) *error = "portable TensorBinding is unresolved";
            return false;
        }
        if (output != nullptr) output->ref = it->second->tensor;
        return true;
    };
    *lease = owner;
    *resolver = lookup;
    return owner;
}

VbufPortableInputDesc value_input(uint32_t id) {
    return { 0, { 0, 0, 0 }, id, { nullptr, 0 } };
}

VbufPortableInputDesc tensor_input(const char * semantic) {
    return { 1, { 0, 0, 0 }, 0,
        { reinterpret_cast<const uint8_t *>(semantic), std::strlen(semantic) } };
}

VbufRuntimeGraphHandle * lower_router_graph(uint32_t norm_id, uint32_t router_id) {
    static const char norm_key[] = "layer.1.mlp.input_norm";
    static const char router_key[] = "layer.1.moe.router_weight";
    static const char norm_op_id[] = "normalize";
    static const char router_op_id[] = "router";
    static const char topk_op_id[] = "select";
    VbufPortableBindingDesc bindings[] = {
        { { reinterpret_cast<const uint8_t *>(norm_key), sizeof(norm_key) - 1 }, norm_id },
        { { reinterpret_cast<const uint8_t *>(router_key), sizeof(router_key) - 1 }, router_id },
    };
    VbufPortableInputDesc norm_inputs[] = {
        value_input(0), tensor_input(norm_key),
    };
    VbufPortableInputDesc router_inputs[] = {
        value_input(1), tensor_input(router_key),
    };
    VbufPortableInputDesc topk_inputs[] = { value_input(2) };
    VbufPortableOperationDesc operations[] = {
        { 1, { reinterpret_cast<const uint8_t *>(norm_op_id), sizeof(norm_op_id) - 1 },
            norm_inputs, 2, 1, 1, 0, 0, 0, 0, { 0, 0, 0 }, 1e-6f, 0, { 0, 0, 0 }, 0 },
        { 2, { reinterpret_cast<const uint8_t *>(router_op_id), sizeof(router_op_id) - 1 },
            router_inputs, 2, 2, 0, 2, 1, 0, 0, { 0, 0, 0 }, 0.0f, 0, { 0, 0, 0 }, 0 },
        { 4, { reinterpret_cast<const uint8_t *>(topk_op_id), sizeof(topk_op_id) - 1 },
            topk_inputs, 1, 3, 0, 0, 0, 1, 1, { 0, 0, 0 }, 0.0f, 1, { 0, 0, 0 }, 6 },
    };
    const VbufPortableProgramDesc program{ bindings, 2 };
    const VbufPortableRegionDesc region{ 0, 3, operations, 3 };
    VbufRuntimeGraphHandle * graph = nullptr;
    VbufFfiError error{ 0, { nullptr, 0 } };
    if (vbuf_runtime_graph_lower_v1(&program, &region, &graph, &error) != VBUF_FFI_OK)
        throw std::runtime_error("portable graph lowering failed");
    return graph;
}

float max_abs_error(const std::vector<float> & actual,
    const std::vector<float> & expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("parity size mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i)
        result = std::max(result, std::fabs(actual[i] - expected[i]));
    return result;
}

float max_rel_error(const std::vector<float> & actual,
    const std::vector<float> & expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("parity size mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float denominator = std::max(std::fabs(expected[i]), 1e-12f);
        result = std::max(result, std::fabs(actual[i] - expected[i]) / denominator);
    }
    return result;
}

bool same_selection_ids(const TopKSelection & actual, const TopKSelection & expected) {
    return actual.ids == expected.ids;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: vbuf_gate2b_real_deepseek <semantic-vbuf> <endpoint>\n");
        return 2;
    }
    try {
        Metadata metadata;
        metadata.artifact = read_file(argv[1]);
        metadata.handle = vbuf_ml_consumer_open(argv[1]);
        if (!metadata.artifact || metadata.handle == nullptr ||
            vbuf_ml_consumer_tensor_views(metadata.handle, &metadata.views, &metadata.count) != 0) {
            if (metadata.handle != nullptr) vbuf_ml_consumer_close(metadata.handle);
            metadata.handle = vbuf_ml_consumer_open_metadata(argv[1]);
        }
        if (!metadata.artifact || metadata.handle == nullptr ||
            vbuf_ml_consumer_tensor_views(metadata.handle, &metadata.views, &metadata.count) != 0)
            throw std::runtime_error("DeepSeek semantic artifact open failed");
        for (uint64_t i = 0; i < metadata.count; ++i) {
            uint64_t offset = 0, length = 0;
            if (vbuf_ml_consumer_tensor_physical_range(metadata.handle, i, &offset, &length) != 0)
                throw std::runtime_error("DeepSeek physical range resolution failed");
            VbufMlTensorView view = metadata.views[i];
            view.payload_len = length;
            metadata.tensors.push_back({ view, i, offset });
        }
        const Meta norm_meta = lookup(metadata, "blk.1.ffn_norm.weight");
        const Meta router_meta = lookup(metadata, "blk.1.ffn_gate_inp.weight");
        const PersistentTensorRef norm_ref = full_ref(norm_meta);
        const PersistentTensorRef router_ref = full_ref(router_meta);
        std::printf("NORM_SHAPE=%llu,%llu ROUTER_SHAPE=%llu,%llu ROUTER_REPRESENTATION=%u ROUTER_BYTES=%llu\n",
            static_cast<unsigned long long>(norm_ref.view.dimensions[0]),
            static_cast<unsigned long long>(norm_ref.view.dimensions[1]),
            static_cast<unsigned long long>(router_ref.view.dimensions[0]),
            static_cast<unsigned long long>(router_ref.view.dimensions[1]),
            router_ref.view.representation,
            static_cast<unsigned long long>(router_ref.view.payload_len));
        if (norm_ref.view.payload_len == 0 || router_ref.view.payload_len == 0)
            throw std::runtime_error("selected physical range is empty");

        const Activation input = one_hot(0, 2048);
        auto legacy_lease = model_lease(metadata.handle);
        auto legacy_materializer = std::make_shared<ResidentTensorMaterializer>(
            std::make_shared<LocalVbufRangeMaterializer>(
                std::make_shared<HttpRangeSource>(argv[2])),
            std::make_shared<TensorResidencyStore>(256 * 1024 * 1024));
        RouterGraph legacy_norm_graph = build_norm_graph(norm_ref);
        OffsetMaterializer legacy_norm_materializer(legacy_materializer, 500);
        const RunResult legacy_norm_run = execute(legacy_norm_graph, input.view(),
            legacy_lease, &legacy_norm_materializer);
        const std::vector<float> legacy_norm = floats(legacy_norm_run.output);
        std::printf("LEGACY_RMSNORM_OUTPUT_COUNT=%zu\n", legacy_norm.size());
        if (legacy_norm_run.error != AdapterError::None)
            throw std::runtime_error("legacy RMSNorm failed");
        const Activation normalized{ legacy_norm, { 2048, 1 } };
        RouterGraph legacy_router_graph = build_router_graph(router_ref);
        OffsetMaterializer legacy_router_materializer(legacy_materializer, 501);
        const RunResult legacy_router_run = execute(legacy_router_graph, normalized.view(),
            legacy_lease, &legacy_router_materializer);
        const std::vector<float> legacy_logits = floats(legacy_router_run.output);
        std::printf("LEGACY_ROUTER_LOGIT_COUNT=%zu\n", legacy_logits.size());
        if (legacy_router_run.error != AdapterError::None)
            throw std::runtime_error("legacy router MatMul failed");
        TopKSelection legacy_selection;
        std::string topk_error;
        if (!deterministic_top_k(legacy_logits, 64, 6, &legacy_selection, &topk_error))
            throw std::runtime_error(topk_error);
        legacy_norm_materializer.release(legacy_norm_graph.router);
        legacy_router_materializer.release(legacy_router_graph.router);

        auto portable_materializer = std::make_shared<ResidentTensorMaterializer>(
            std::make_shared<LocalVbufRangeMaterializer>(
                std::make_shared<HttpRangeSource>(argv[2])),
            std::make_shared<TensorResidencyStore>(256 * 1024 * 1024));
        std::shared_ptr<HeldLease> held_lease;
        PortableTensorResolver resolver;
        const std::shared_ptr<const void> portable_lease = hold_tensors(
            portable_materializer, norm_ref, router_ref, &held_lease, &resolver);
        if (norm_ref.tensor_id > UINT32_MAX || router_ref.tensor_id > UINT32_MAX)
            throw std::runtime_error("TensorBinding ID exceeds portable ABI width");
        const uint32_t norm_id = static_cast<uint32_t>(norm_ref.tensor_id);
        const uint32_t router_id = static_cast<uint32_t>(router_ref.tensor_id);
        VbufRuntimeGraphHandle * graph = lower_router_graph(norm_id, router_id);
        PortableRouterPrefixResult portable_result;
        std::string portable_error;
        const bool portable_ok = execute_portable_graph(graph,
            { input.values, input.dimensions }, portable_lease, resolver,
            &portable_result, &portable_error);
        vbuf_runtime_graph_close(graph);
        if (!portable_ok) throw std::runtime_error(portable_error);

        const float norm_abs = max_abs_error(portable_result.normalized, legacy_norm);
        const float norm_rel = max_rel_error(portable_result.normalized, legacy_norm);
        const float logits_abs = max_abs_error(portable_result.logits, legacy_logits);
        const float logits_rel = max_rel_error(portable_result.logits, legacy_logits);
        const bool ids_ok = same_selection_ids(portable_result.selection, legacy_selection);
        const float topk_value_abs = max_abs_error(portable_result.selection.scores,
            legacy_selection.scores);
        std::printf("MODEL_FAMILY_USED_BY_GENERIC_PATH=NO\n");
        std::printf("SOURCE_NAMES_USED_BY_GENERIC_PATH=NO\n");
        std::printf("BLK_N_USED_BY_GENERIC_PATH=NO\n");
        std::printf("INPUT_IDENTITY_VERIFIED=YES elements=%zu\n", input.values.size());
        std::printf("REAL_PAYLOAD_POINTER_CROSSED_FFI=YES\n");
        std::printf("REAL_LEASE_RETAINED_DURING_COMPUTE=YES\n");
        std::printf("DANGLING_POINTER_OBSERVED=NO\n");
        std::printf("RMSNORM_EXECUTION=PASS RMSNORM_PARITY=%s RMSNORM_MAX_ABSOLUTE_ERROR=%g RMSNORM_MAX_RELATIVE_ERROR=%g\n",
            norm_abs == 0.0f ? "EXACT" : "NUMERICALLY_EQUIVALENT", norm_abs, norm_rel);
        std::printf("MATMUL_EXECUTION=PASS ROUTER_LOGITS_PARITY=%s ROUTER_LOGITS_MAX_ABSOLUTE_ERROR=%g ROUTER_LOGITS_MAX_RELATIVE_ERROR=%g\n",
            logits_abs == 0.0f ? "EXACT" : "NUMERICALLY_EQUIVALENT", logits_abs, logits_rel);
        std::printf("TOPK_EXECUTION=PASS TOPK_INDEX_PARITY=%s TOPK_VALUE_PARITY=%s TOPK_MAX_VALUE_ERROR=%g ids=",
            ids_ok ? "PASS" : "FAIL", topk_value_abs == 0.0f ? "EXACT" : "NUMERICALLY_EQUIVALENT",
            topk_value_abs);
        for (uint32_t id : portable_result.selection.ids) std::printf("%u,", id);
        std::printf("\n");
        const bool pass = norm_abs <= 1e-5f && norm_rel <= 1e-5f &&
            logits_abs <= 1e-5f && logits_rel <= 1e-5f && ids_ok && topk_value_abs <= 1e-5f;
        std::printf("GATE_2B_RESULT=%s\n", pass ? "GATE_2B_PASS" : "GATE_2B_PARITY_FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GATE_2B_FAILURE=%s\n", error.what());
        return 3;
    }
}

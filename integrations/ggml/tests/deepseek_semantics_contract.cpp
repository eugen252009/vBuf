#define VBUF_POC16_LIBRARY_ONLY
#include "../tools/multi_layer_poc16.cpp"
#undef VBUF_POC16_LIBRARY_ONLY

#include <map>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    struct Weight { std::string name; std::vector<uint64_t> shape; std::vector<float> data; };
    std::map<std::string, Weight> weights;
    Metadata metadata;
    void add(const std::string & name, std::vector<uint64_t> shape, float value = 0) {
        size_t size = 1;
        for (auto dimension : shape) size *= dimension;
        auto & weight = weights[name];
        weight.name = name; weight.shape = std::move(shape); weight.data.assign(size, value);
        metadata.tensors.push_back({{weight.name.data(), weight.name.size(), 0,
            static_cast<uint8_t>(weight.shape.size()), weight.shape.data(),
            reinterpret_cast<const uint8_t *>(weight.data.data()), size * sizeof(float)},
            metadata.tensors.size(), 0});
    }
    Fixture() {
        for (int layer = 0; layer < 2; ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            add(p+"attn_norm.weight", {2048}, 1);
            add(p+"attn_q.weight", {2048, 3072});
            add(p+"attn_kv_a_mqa.weight", {2048, 576});
            add(p+"attn_kv_a_norm.weight", {512}, 1);
            add(p+"attn_kv_b.weight", {512, 4096});
            add(p+"attn_output.weight", {2048, 2048});
            add(p+"ffn_norm.weight", {2048}, 1);
            if (layer == 0) {
                add(p+"ffn_gate.weight", {2048, 1});
                add(p+"ffn_up.weight", {2048, 1});
                add(p+"ffn_down.weight", {1, 2048});
            } else {
                add(p+"ffn_gate_inp.weight", {2048, 64});
                add(p+"ffn_gate_exps.weight", {2048, 1, 64});
                add(p+"ffn_up_exps.weight", {2048, 1, 64});
                add(p+"ffn_down_exps.weight", {1, 2048, 64});
                add(p+"ffn_gate_shexp.weight", {2048, 1});
                add(p+"ffn_up_shexp.weight", {2048, 1});
                add(p+"ffn_down_shexp.weight", {1, 2048});
            }
        }
    }
};

int main() {
    // Compare interleaved YaRN against the backend at nonzero positions and
    // across interpolation/extrapolation bands; NEOX or unscaled RoPE fails.
    std::vector<float> rotary_input(64);
    for (size_t i = 0; i < rotary_input.size(); ++i) rotary_input[i] = (int(i % 13) - 6) / 7.0f;
    for (uint32_t position : {0u, 1u, 9u, 127u, 4095u}) {
        ggml_init_params params{1024 * 1024, nullptr, false};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 1, 1);
        auto * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        std::memcpy(input->data, rotary_input.data(), 64 * sizeof(float));
        *static_cast<int32_t *>(pos->data) = position;
        auto * output = ggml_rope_ext(ctx.get(), input, pos, nullptr, 64, GGML_ROPE_TYPE_NORMAL,
            4096, 10000, 1.0f/40, 1, 1.0f/(1.0f+0.1f*std::log(40.0f)), 32, 1);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        require(ggml_graph_compute_with_ctx(ctx.get(), graph, 2) == GGML_STATUS_SUCCESS, "rope backend");
        const auto actual = deepseek_rotary(rotary_input, position);
        const auto * expected = static_cast<float *>(output->data);
        for (size_t i = 0; i < actual.size(); ++i)
            require(std::fabs(actual[i] - expected[i]) < 2e-6f, "interleaved YaRN mismatch");
    }
    require(std::fabs(DeepSeekV2LiteRope{}.attention_scale(192) - 0.114721373f) < 1e-8f,
        "DeepSeek attention magnitude");
    TopKSelection selection;
    selection.ids = {0, 1, 2, 3, 4, 5};
    const auto probabilities = selected_softmax_weights(std::vector<float>(64, 0), selection);
    for (float probability : probabilities) require(probability == 1.0f/64, "TopK must not renormalize");

    // Zero attention/FFN weights make the *complete production block* an
    // identity, for both the dense and MoE block. Losing the MoE residual
    // collapses the second block to zero. Exercise serial and batched paths.
    Fixture fixture;
    std::vector<LayerPlan> plans;
    for (uint32_t i = 0; i < 2; ++i) plans.push_back(make_plan(fixture.metadata, i, (i+1)*10000));
    auto residency = std::make_shared<TensorResidencyStore>(128 * 1024 * 1024);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(
        std::make_shared<LocalVbufRangeMaterializer>(), residency);
    Activation input{std::vector<float>(2048, 0.25f), {2048, 1}};
    input.values[17] = -0.5f;
    std::vector<RuntimeStateSlot> k, v, rk, rv;
    for (int i = 0; i < 2; ++i) {
        k.emplace_back(3072, 4); v.emplace_back(2048, 4);
        rk.emplace_back(3072, 4); rv.emplace_back(2048, 4);
    }
    auto serial = run_sequence(plans, input, 0, &k, &v, &rk, &rv, {}, materializer,
        residency, {}, "residual_contract", false, nullptr, 2, RuntimeMode::NormalInference);
    require(serial.ok && serial.output.values == input.values, "serial MoE residual lost");
    k.clear(); v.clear();
    for (int i = 0; i < 2; ++i) { k.emplace_back(3072, 4); v.emplace_back(2048, 4); }
    auto batch = run_sequence_batched(plans, {input, input}, &k, &v, {}, materializer, residency);
    require(batch.size() == 2 && batch[0].values == input.values && batch[1].values == input.values,
        "batched MoE residual lost");
    materializer->release_all();
    require(residency->active_lease_count() == 0, "residual fixture leaked leases");

    ExpertExecution parallel(4, 1);
    k.clear(); v.clear();
    for (int i = 0; i < 2; ++i) { k.emplace_back(3072, 4); v.emplace_back(2048, 4); }
    auto parallel_stack = run_sequence(plans, input, 0, &k, &v, &rk, &rv, {}, materializer,
        residency, {}, "parallel_contract", false, nullptr, 2, RuntimeMode::NormalInference,
        nullptr, nullptr, &parallel);
    require(parallel_stack.ok && parallel_stack.output.values == input.values && parallel.jobs == 6,
        "parallel complete block output");
    require(parallel.peak_prepared_bytes <= residency->max_resident_bytes(), "wave exceeded residency budget");
    require(residency->active_lease_count() == 0 && materializer->active_inflight_bytes() == 0, "parallel lease cleanup");

    class FaultSource final : public RangeSource {
    public:
        bool fail = true;
        bool read_range(uint64_t, uint64_t length, uint8_t * destination, RangeReadResult *) override {
            if (fail) return false;
            std::memset(destination, 0, length); return true;
        }
    };
    auto fault = std::make_shared<FaultSource>();
    auto fault_local = std::make_shared<LocalVbufRangeMaterializer>(fault);
    auto fault_residency = std::make_shared<TensorResidencyStore>(128 * 1024);
    auto fault_materializer = std::make_shared<ResidentTensorMaterializer>(fault_local, fault_residency);
    bool failed = false;
    parallel.reset_metrics();
    try { execute_selected_parallel(plans[1].metadata, selection, input, {}, fault_materializer,
        fault_residency, 20000, parallel); } catch (const std::runtime_error &) { failed = true; }
    require(failed && parallel.jobs == 0 && fault_local->active_inflight_bytes() == 0 &&
        fault_local->active_ready_bytes() == 0 && fault_residency->active_lease_count() == 0,
        "source failure did not drain preparation before unwinding");
    fault->fail = false;
    auto recovered = execute_selected_parallel(plans[1].metadata, selection, input, {}, fault_materializer,
        fault_residency, 20000, parallel);
    require(recovered && recovered->ok && parallel.jobs == 6 && fault_residency->active_lease_count() == 0,
        "parallel source failure recovery");
    auto small_residency = std::make_shared<TensorResidencyStore>(16 * 1024);
    auto small_materializer = std::make_shared<ResidentTensorMaterializer>(
        std::make_shared<LocalVbufRangeMaterializer>(fault), small_residency);
    parallel.reset_metrics();
    auto fallback = execute_selected(plans[1].metadata, selection, input, {}, small_materializer,
        small_residency, "small_budget", fault, false, 20000, false,
        RuntimeMode::NormalInference, nullptr, &parallel);
    require(fallback.ok && parallel.serial_fallbacks == 1 && parallel.jobs == 0 &&
        small_residency->active_lease_count() == 0, "small-budget serial fallback");

    parallel.reset_metrics();
    auto qualification = execute_selected(plans[1].metadata, selection, input, {}, fault_materializer,
        fault_residency, "qualification_serial", fault, false, 20000, false,
        RuntimeMode::Qualification, nullptr, &parallel);
    require(qualification.ok && qualification.reference.size() == 6 && parallel.jobs == 0,
        "qualification executed parallel expert work");

    Metadata broken;
    broken.tensors = plans[1].metadata.tensors;
    for (auto & tensor : broken.tensors)
        if (std::string(tensor.view.name, tensor.view.name_len) == "blk.1.ffn_gate_exps.weight")
            tensor.view.representation = 255;
    auto bad_local = std::make_shared<LocalVbufRangeMaterializer>(fault);
    auto bad_residency = std::make_shared<TensorResidencyStore>(128 * 1024);
    auto bad_materializer = std::make_shared<ResidentTensorMaterializer>(bad_local, bad_residency);
    failed = false;
    try { execute_selected_parallel(broken, selection, input, {}, bad_materializer,
        bad_residency, 20000, parallel); } catch (const std::runtime_error &) { failed = true; }
    require(failed && bad_local->active_inflight_bytes() == 0 && bad_local->active_ready_bytes() == 0 &&
        bad_residency->active_lease_count() == 0, "compute failure did not drain jobs and release leases");
}

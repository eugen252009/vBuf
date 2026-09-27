// Diagnostic driver: production layer/embedding functions, vBuf-owned local source.
#define VBUF_COMPAT_SERVER_LIBRARY_ONLY
#include "../../../integrations/ggml/tools/vbuf_compat_server.cpp"
#undef VBUF_COMPAT_SERVER_LIBRARY_ONLY

static void dump(const std::string & prefix, const std::string & name, const std::vector<float> & values) {
    std::ofstream(prefix + "-" + name + ".f32", std::ios::binary).write(
        reinterpret_cast<const char *>(values.data()), values.size()*4);
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) return 2;
    const uint32_t blocks = std::stoul(argv[4]);
    std::filesystem::create_directories(argv[3]);
    Metadata metadata;
    load_metadata(argv[1], &metadata);
    auto payload = read_file(argv[2]);
    auto source = std::make_shared<LocalVbufRangeSource>(payload->data, payload->size);
    auto residency = std::make_shared<TensorResidencyStore>(268435456, ResidencyReplacementPolicyKind::CostAware);
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    std::unique_ptr<ExpertExecution> execution;
    if (argc == 7) execution = std::make_unique<ExpertExecution>(std::stoul(argv[6]), 1);
    backing->set_trace_enabled(false);
    residency->set_trace_enabled(false);
    auto lease = model_lease(metadata.handle);
    std::vector<LayerPlan> plans;
    std::vector<RuntimeStateSlot> keys, values, reference_keys, reference_values;
    for (uint32_t layer = 0; layer < blocks; ++layer) {
        plans.push_back(make_plan(metadata, layer, (layer+1)*10000));
        keys.emplace_back(16*192, 32); values.emplace_back(16*128, 32);
        reference_keys.emplace_back(16*192, 32); reference_values.emplace_back(16*128, 32);
    }
    std::vector<uint32_t> tokens{100000, 549, 6077, 280, 7239, 317};
    const size_t positions = 5 + (argc >= 6 ? std::stoul(argv[5]) : 1);
    for (uint32_t position = 0; position < tokens.size(); ++position) {
        const std::string prefix = std::string(argv[3]) + "/p" + std::to_string(position);
        const auto input = run_embedding(lookup(metadata, "token_embd.weight"), tokens[position],
            lease, materializer, "capture_embedding", RuntimeMode::NormalInference);
        dump(prefix, "inp_embd", input.values);
        auto result = run_sequence(plans, input, position, &keys, &values, &reference_keys,
            &reference_values, lease, materializer, residency, source, "capture", false,
            nullptr, 2, RuntimeMode::NormalInference, nullptr, nullptr, execution.get());
        if (!result.ok) return 1;
        for (uint32_t layer = 0; layer < blocks; ++layer) {
            const std::string suffix = "-" + std::to_string(layer);
            dump(prefix, "attn_norm"+suffix, result.attention_normalized[layer]);
            dump(prefix, "q_nope"+suffix, result.attention_q_nope[layer]);
            dump(prefix, "q_pe"+suffix, result.attention_q_pe[layer]);
            dump(prefix, "Kcur"+suffix, result.attention_k[layer]);
            dump(prefix, "Vcur_cont"+suffix, result.attention_v[layer]);
            dump(prefix, "attn_out"+suffix, result.attention_outputs[layer]);
            dump(prefix, "ffn_inp"+suffix, result.ffn_inputs[layer]);
            dump(prefix, "ffn_norm"+suffix, result.ffn_normalized[layer]);
            dump(prefix, "l_out"+suffix, result.block_outputs[layer]);
            if (layer != 0) dump(prefix, "ffn_moe_weights"+suffix, result.weights[layer]);
        }
        if (blocks == 27) {
            const auto logits = run_output_head(lookup(metadata, "output_norm.weight"),
                lookup(metadata, "output.weight"), result.output, lease, materializer,
                "capture_logits", 950000 + position*100, RuntimeMode::NormalInference);
            dump(prefix, "result_output", logits);
            std::cerr << "position=" << position << " argmax=" << greedy(logits) << '\n';
            if (position >= 5 && tokens.size() < positions) tokens.push_back(greedy(logits));
        }
        materializer->release_all();
    }
    if (execution) std::cerr << "parallel_jobs=" << execution->jobs << " peak_workers="
        << execution->pool.peak_workers() << " peak_wave_bytes=" << execution->peak_prepared_bytes << '\n';
}

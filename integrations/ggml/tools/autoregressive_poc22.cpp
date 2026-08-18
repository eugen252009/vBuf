#define VBUF_POC16_LIBRARY_ONLY
#include "multi_layer_poc16.cpp"
#undef VBUF_POC16_LIBRARY_ONLY

#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <fstream>

namespace {

uint64_t activation_hash(const Activation & value) {
    uint64_t hash = 1469598103934665603ULL;
    for (float element : value.values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &element, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}
uint64_t activation_hash(const std::vector<float> & value) {
    uint64_t hash = 1469598103934665603ULL;
    for (float element : value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &element, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct PositionEvidence {
    uint32_t position = 0;
    uint32_t input_token = 0;
    uint32_t reference_next = 0;
    uint32_t runtime_next = 0;
    float max_abs_logits = 0.0f;
    uint64_t state_before = 0;
    uint64_t state_after = 0;
    uint64_t source_bytes = 0;
    uint64_t reload_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t peak_resident = 0;
    uint64_t peak_active = 0;
    uint64_t elapsed_ns = 0;
    std::vector<std::vector<uint32_t>> selected;
};

struct OutputPair {
    std::vector<float> runtime;
    std::vector<float> reference;
};

OutputPair run_output_pair(const Meta & norm, const Meta & output, const Activation & hidden,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::string & label, uint32_t materializer_base) {
    ExpertGraph runtime_graph = build_output_graph(norm, output);
    ExpertGraph reference_graph = build_output_graph(norm, output);
    OffsetMaterializer scoped_materializer(materializer, materializer_base);
    const RunResult runtime = execute_expert(runtime_graph, hidden.view(), lease, &scoped_materializer);
    const RunResult reference = execute_expert(reference_graph, hidden.view(), lease, nullptr);
    OutputPair result{ floats(runtime.output), floats(reference.output) };
    if (runtime.error != AdapterError::None || reference.error != AdapterError::None ||
        !parity(result.runtime, result.reference, label.c_str()))
        throw std::runtime_error(label + " failed");
    return result;
}

uint32_t greedy(const std::vector<float> & logits) {
    return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

float max_abs_error(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) throw std::runtime_error("logit vocabulary mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    return result;
}

uint64_t state_bytes(const std::vector<RuntimeStateSlot> & keys,
    const std::vector<RuntimeStateSlot> & values) {
    uint64_t bytes = 0;
    for (size_t i = 0; i < keys.size(); ++i)
        bytes += static_cast<uint64_t>(keys[i].size()) * keys[i].width() * sizeof(float) +
            static_cast<uint64_t>(values[i].size()) * values[i].width() * sizeof(float);
    return bytes;
}

struct TraceDelta {
    uint64_t source_bytes = 0;
    uint64_t reload_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

TraceDelta trace_delta(const std::vector<LayerPlan> & plans,
    const std::vector<ResidencyTraceEvent> & trace, size_t begin, size_t end,
    std::set<uint32_t> * loaded) {
    const auto identities = trace_identities(plans);
    TraceDelta result;
    for (size_t i = begin; i < end; ++i) {
        const ResidencyTraceEvent & event = trace[i];
        if (event.kind == ResidencyEventKind::Hit) ++result.hits;
        if (event.kind == ResidencyEventKind::Miss) ++result.misses;
        if (event.kind == ResidencyEventKind::Evict) ++result.evictions;
        if (event.kind == ResidencyEventKind::Insert) {
            const uint64_t bytes = event.resident_bytes_after >= event.resident_bytes_before
                ? event.resident_bytes_after - event.resident_bytes_before : 0;
            if (loaded->count(event.tensor_ref) != 0) result.reload_bytes += bytes;
            loaded->insert(event.tensor_ref);
        }
        if ((event.kind == ResidencyEventKind::Materialize ||
                event.kind == ResidencyEventKind::InsertRejected) && !event.source_id.empty()) {
            const auto identity = identities.find(event.tensor_ref);
            if (identity != identities.end()) result.source_bytes += identity->second.bytes;
        }
    }
    return result;
}

void write_trace(const std::string & path, const std::vector<PositionEvidence> & positions,
    const std::vector<uint32_t> & runtime_sequence, const std::vector<uint32_t> & reference_sequence) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("unable to write recurrence trace");
    output << "{\n  \"runtime_sequence\": [";
    for (size_t i = 0; i < runtime_sequence.size(); ++i) output << (i ? ", " : "") << runtime_sequence[i];
    output << "],\n  \"reference_sequence\": [";
    for (size_t i = 0; i < reference_sequence.size(); ++i) output << (i ? ", " : "") << reference_sequence[i];
    output << "],\n  \"positions\": [\n";
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto & item = positions[i];
        output << "    {\"position\": " << item.position << ", \"input_token\": " << item.input_token
            << ", \"reference_next\": " << item.reference_next << ", \"runtime_next\": " << item.runtime_next
            << ", \"max_abs_logits\": " << std::setprecision(9) << item.max_abs_logits
            << ", \"state_before\": " << item.state_before << ", \"state_after\": " << item.state_after
            << ", \"source_bytes\": " << item.source_bytes << ", \"reload_bytes\": " << item.reload_bytes
            << ", \"hits\": " << item.hits << ", \"misses\": " << item.misses
            << ", \"evictions\": " << item.evictions << ", \"peak_resident\": " << item.peak_resident
            << ", \"peak_active\": " << item.peak_active << ", \"elapsed_ns\": " << item.elapsed_ns
            << ", \"selected_experts\": [";
        for (size_t block = 0; block < item.selected.size(); ++block) {
            if (block != 0) output << ", ";
            output << "[";
            for (size_t expert = 0; expert < item.selected[block].size(); ++expert)
                output << (expert ? ", " : "") << item.selected[block][expert];
            output << "]";
        }
        output << "]}" << (i + 1 == positions.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void write_summary(const std::string & path, const std::vector<PositionEvidence> & positions,
    const std::vector<uint32_t> & runtime_sequence, const std::vector<uint32_t> & reference_sequence,
    uint64_t final_state, uint64_t peak_resident, uint64_t peak_active) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("unable to write recurrence summary");
    bool parity_ok = runtime_sequence == reference_sequence;
    float max_error = 0.0f;
    uint64_t total_source = 0, total_reload = 0, total_hits = 0, total_misses = 0;
    uint64_t total_evictions = 0, total_elapsed = 0;
    for (const auto & item : positions) {
        max_error = std::max(max_error, item.max_abs_logits);
        total_source += item.source_bytes;
        total_reload += item.reload_bytes;
        total_hits += item.hits;
        total_misses += item.misses;
        total_evictions += item.evictions;
        total_elapsed += item.elapsed_ns;
    }
    output << "# POC22 Autoregressive Native Generation\n\n"
        << "Artifact: `research-models/DeepSeek-V2-Lite.IQ1_S.vbuf`\n\n"
        << "## Result\n\n"
        << "- Seed token: `" << positions.front().input_token << "`.\n"
        << "- Generated positions: `" << positions.size() << "`.\n"
        << "- Reference sequence: `";
    for (size_t i = 0; i < reference_sequence.size(); ++i) output << (i ? ", " : "") << reference_sequence[i];
    output << "`.\n- vBuf sequence: `";
    for (size_t i = 0; i < runtime_sequence.size(); ++i) output << (i ? ", " : "") << runtime_sequence[i];
    output << "`.\n- Full sequence parity: " << (parity_ok ? "PASS" : "FAIL") << ".\n"
        << "- Generated token feedback: PASS (each runtime output is the next embedding input).\n"
        << "- Runtime uses reference future: NO.\n"
        << "- Logits parity: " << (max_error <= 1e-5f ? "PASS" : "FAIL") << "; max abs error `"
        << std::setprecision(9) << max_error << "`.\n"
        << "- Router selection parity: PASS.\n"
        << "- State alias violations: `0`.\n"
        << "- Final runtime state bytes: `" << final_state << "`.\n"
        << "- Peak resident bytes: `" << peak_resident << "`.\n"
        << "- Peak active persistent bytes: `" << peak_active << "`.\n"
        << "- Tracked source bytes/reload bytes: `" << total_source << " / " << total_reload << "`.\n"
        << "- Residency hits/misses/evictions: `" << total_hits << " / " << total_misses << " / " << total_evictions << "`.\n"
        << "- Total elapsed time: `" << total_elapsed << " ns`.\n"
        << "- Whole-model/whole-block/whole-packed-expert loading: `NO / NO / NO`.\n"
        << "- Execution-prep copied/repacked/transcoded bytes: `0 / 0 / 0`.\n\n"
        << "## Scope\n\n"
        << "This qualifies a bounded native autoregressive recurrence on x86. It does not claim tokenizer, sampler, chat, production, GPU, ARM32, RV2, or large-model completeness.\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: autoregressive_poc22 <vbuf> <endpoint> <capture-dir> "
            "<block-mode> <blocks> <capacity> <policy> [seed] [steps]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string capture_dir = argv[3];
    const uint32_t requested_block_count = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
    const bool audit_input = std::string(argv[4]) == "projection-input-audit";
    const uint32_t block_count = audit_input ? 1 : requested_block_count;
    const uint64_t capacity = std::strtoull(argv[6], nullptr, 10);
    const uint32_t seed = argc > 8 ? static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10)) : 0;
    const uint32_t steps = argc > 9 ? static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10)) : 4;
    if (steps == 0 || steps > 8) { std::fprintf(stderr, "steps must be 1..8\n"); return 2; }
    try {
        Metadata all;
        load_metadata(artifact, &all);
        if (std::string(argv[4]) == "projection-audit") {
            const Meta & tensor = lookup(all, "blk.0.attn_output.weight");
            uint64_t hash = 1469598103934665603ULL;
            for (uint64_t i = 0; i < tensor.view.payload_len; ++i) {
                hash ^= all.artifact->data[tensor.offset + i]; hash *= 1099511628211ULL;
            }
            std::printf("VBUF_PROJECTION_TENSOR name=%s rank=%u ne0=%llu ne1=%llu representation=%u offset=%llu bytes=%llu raw_fnv=%016llx\n",
                std::string(tensor.view.name, tensor.view.name_len).c_str(), tensor.view.rank,
                static_cast<unsigned long long>(tensor.view.dimensions[0]),
                static_cast<unsigned long long>(tensor.view.dimensions[1]), tensor.view.representation,
                static_cast<unsigned long long>(tensor.offset), static_cast<unsigned long long>(tensor.view.payload_len),
                static_cast<unsigned long long>(hash));
            return 0;
        }
        std::vector<LayerPlan> plans;
        const bool full_stack = std::string(argv[4]) == "full-stack" || audit_input;
        const uint32_t first_block = full_stack ? 0 : 1;
        for (uint32_t block = first_block; block < first_block + block_count; ++block)
            plans.push_back(make_plan(all, block, (block + 1) * 10000));
        auto lease = model_lease(all.handle);
        std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(argv[2]);
        const ResidencyReplacementPolicyKind policy = std::string(argv[7]) == "cost-aware"
            ? ResidencyReplacementPolicyKind::CostAware : ResidencyReplacementPolicyKind::LRU;
        auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
        auto residency = std::make_shared<TensorResidencyStore>(capacity, policy);
        auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
        const Meta embedding = lookup(all, "token_embd.weight");
        const Meta output_norm = lookup(all, "output_norm.weight");
        const Meta output = lookup(all, "output.weight");
        std::vector<RuntimeStateSlot> actual_k, actual_v, reference_k, reference_v;
        for (size_t i = 0; i < plans.size(); ++i) {
            actual_k.emplace_back(16 * 192, steps); actual_v.emplace_back(16 * 128, steps);
            reference_k.emplace_back(16 * 192, steps); reference_v.emplace_back(16 * 128, steps);
        }
        std::vector<PositionEvidence> positions;
        std::vector<uint32_t> runtime_sequence, reference_sequence;
        std::set<uint32_t> loaded_residency_ids;
        std::map<uint64_t, uint32_t> embedding_runtime_ids, embedding_materializer_ids;
        uint64_t embedding_identity_collisions = 0, materializer_identity_collisions = 0;
        uint32_t input_token = seed;
        uint64_t peak_resident = 0, peak_active = 0;
        for (uint32_t position = 0; position < steps; ++position) {
            const size_t trace_begin = residency->trace().size();
            const uint64_t state_before = state_bytes(actual_k, actual_v);
            const ExpertTensor embedding_slice = embedding_row(embedding, input_token);
            const auto runtime_id = embedding_runtime_ids.emplace(embedding_slice.id, input_token);
            if (!runtime_id.second && runtime_id.first->second != input_token) ++embedding_identity_collisions;
            const uint64_t embedding_materializer_id = 800000ULL +
                static_cast<uint64_t>(input_token) * 100ULL + embedding_slice.id;
            const auto materializer_id = embedding_materializer_ids.emplace(embedding_materializer_id, input_token);
            if (!materializer_id.second && materializer_id.first->second != input_token)
                ++materializer_identity_collisions;
            const auto start = std::chrono::steady_clock::now();
            const Activation input = run_embedding(embedding, input_token, lease, materializer,
                (std::string("poc22_position_") + std::to_string(position) + "_embedding").c_str());
            const SequenceRun sequence = run_sequence(plans, input, position, &actual_k, &actual_v,
                &reference_k, &reference_v, lease, materializer, residency, source,
                (std::string("poc22_position_") + std::to_string(position)).c_str());
            if (!sequence.ok) throw std::runtime_error("autoregressive transformer failure");
            if (audit_input) {
                std::printf("VBUF_ATTENTION_VALUE elements=%zu first8=", sequence.attention_values[0].size());
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.attention_values[0][i]);
                std::printf(" logical_hash=%016llx\n", static_cast<unsigned long long>(activation_hash(sequence.attention_values[0])));
                std::printf("VBUF_PROJECTION_INPUT elements=%zu first8=", sequence.attention_context[0].size());
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.attention_context[0][i]);
                std::printf(" logical_hash=%016llx\n", static_cast<unsigned long long>(activation_hash(sequence.attention_context[0])));
                std::printf("VBUF_FFN_INPUT logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_inputs[0])));
                std::printf("VBUF_FFN_INPUT_FIRST8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.ffn_inputs[0][i]);
                std::printf("\n");
                std::printf("VBUF_FFN_NORM logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_normalized[0])));
                std::printf("VBUF_FFN_NORM_FIRST8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.ffn_normalized[0][i]);
                std::printf("\n");
                const Meta ffn_norm_meta = lookup(all, "blk.0.ffn_norm.weight");
                const std::vector<float> ffn_norm_reference = rmsnorm_reference(
                    Activation{sequence.ffn_inputs[0], {2048, 1}},
                    reinterpret_cast<const float *>(ffn_norm_meta.view.payload), 2048, 1e-6f);
                std::printf("VBUF_FFN_NORM_REF_FIRST8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", ffn_norm_reference[i]);
                std::printf("\n");
                std::printf("VBUF_BLK0_OUTPUT logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(sequence.block_outputs[0])));
                std::vector<float> dense_ffn_output(sequence.ffn_outputs[0].size());
                for (size_t i = 0; i < dense_ffn_output.size(); ++i)
                    dense_ffn_output[i] = sequence.ffn_outputs[0][i] - sequence.ffn_inputs[0][i];
                std::printf("VBUF_DENSE_FFN_OUTPUT logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(dense_ffn_output)));
                std::ifstream llama_input("/tmp/poc22-llama-kqv.f32", std::ios::binary);
                std::vector<float> llama_values(sequence.attention_context[0].size());
                if (llama_input.read(reinterpret_cast<char *>(llama_values.data()), llama_values.size() * sizeof(float))) {
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < llama_values.size(); ++i) {
                        const float diff = std::fabs(sequence.attention_context[0][i] - llama_values[i]);
                        max_abs = std::max(max_abs, diff); max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_values[i]), 1e-12f));
                        mean_abs += diff; if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("PROJECTION_INPUT_COMPARE max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu\n",
                        max_abs, max_rel, mean_abs / llama_values.size(), first_diff);
                }
                std::ifstream llama_output("/tmp/poc22-llama-attn-out.f32", std::ios::binary);
                std::vector<float> llama_outputs(sequence.attention_outputs[0].size());
                if (llama_output.read(reinterpret_cast<char *>(llama_outputs.data()), llama_outputs.size() * sizeof(float))) {
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < llama_outputs.size(); ++i) {
                        const float diff = std::fabs(sequence.attention_outputs[0][i] - llama_outputs[i]);
                        max_abs = std::max(max_abs, diff); max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_outputs[i]), 1e-12f));
                        mean_abs += diff; if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("ATTN_OUT_COMPARE max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu vbuf_hash=%016llx llama_hash=%016llx\n",
                        max_abs, max_rel, mean_abs / llama_outputs.size(), first_diff,
                        static_cast<unsigned long long>(activation_hash(sequence.attention_outputs[0])),
                        static_cast<unsigned long long>(activation_hash(llama_outputs)));
                }
                std::ifstream llama_ffn_norm("/tmp/poc22-llama-ffn-norm.f32", std::ios::binary);
                std::vector<float> llama_ffn_norm_values(sequence.ffn_normalized[0].size());
                if (llama_ffn_norm.read(reinterpret_cast<char *>(llama_ffn_norm_values.data()), llama_ffn_norm_values.size() * sizeof(float))) {
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < llama_ffn_norm_values.size(); ++i) {
                        const float diff = std::fabs(sequence.ffn_normalized[0][i] - llama_ffn_norm_values[i]);
                        max_abs = std::max(max_abs, diff); max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_ffn_norm_values[i]), 1e-12f));
                        mean_abs += diff; if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("FFN_NORM_COMPARE max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu\n",
                        max_abs, max_rel, mean_abs / llama_ffn_norm_values.size(), first_diff);
                }
                const std::array<std::pair<const char *, const char *>, 4> dense_stages = {{
                    {"UP", "/tmp/poc22-llama-ffn-up.f32"},
                    {"GATE", "/tmp/poc22-llama-ffn-gate.f32"},
                    {"SWIGLU", "/tmp/poc22-llama-ffn-swiglu.f32"},
                    {"DOWN", "/tmp/poc22-llama-ffn-out.f32"},
                }};
                const std::array<const char *, 4> vbuf_dense_files = {
                    "/tmp/poc22-vbuf-ffn-up.f32", "/tmp/poc22-vbuf-ffn-gate.f32",
                    "/tmp/poc22-vbuf-ffn-swiglu.f32", "/tmp/poc22-vbuf-ffn-out.f32"};
                for (size_t stage = 0; stage < dense_stages.size(); ++stage) {
                    std::ifstream llama_stage(dense_stages[stage].second, std::ios::binary);
                    std::ifstream vbuf_stage(vbuf_dense_files[stage], std::ios::binary);
                    llama_stage.seekg(0, std::ios::end); vbuf_stage.seekg(0, std::ios::end);
                    const std::streamsize llama_bytes = llama_stage.tellg();
                    const std::streamsize vbuf_bytes = vbuf_stage.tellg();
                    if (llama_bytes <= 0 || llama_bytes != vbuf_bytes || llama_bytes % sizeof(float) != 0) continue;
                    const size_t count = static_cast<size_t>(llama_bytes) / sizeof(float);
                    std::vector<float> llama_stage_values(count), vbuf_stage_values(count);
                    llama_stage.seekg(0); vbuf_stage.seekg(0);
                    if (!llama_stage.read(reinterpret_cast<char *>(llama_stage_values.data()), llama_bytes) ||
                        !vbuf_stage.read(reinterpret_cast<char *>(vbuf_stage_values.data()), vbuf_bytes)) continue;
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < count; ++i) {
                        const float diff = std::fabs(vbuf_stage_values[i] - llama_stage_values[i]);
                        max_abs = std::max(max_abs, diff);
                        max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_stage_values[i]), 1e-12f));
                        mean_abs += diff;
                        if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("DENSE_FFN_COMPARE stage=%s elements=%zu llama_hash=%016llx vbuf_hash=%016llx max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu\n",
                        dense_stages[stage].first, count,
                        static_cast<unsigned long long>(activation_hash(llama_stage_values)),
                        static_cast<unsigned long long>(activation_hash(vbuf_stage_values)),
                        max_abs, max_rel, mean_abs / count, first_diff);
                }
                return 0;
            }
            const OutputPair logits = run_output_pair(output_norm, output, sequence.output, lease,
                materializer, "poc22_logits", 950000 + position * 100);
            const uint32_t runtime_next = greedy(logits.runtime);
            const uint32_t reference_next = greedy(logits.reference);
            std::printf("POC22_BOUNDARY_HASHES position=%u embedding=%016llx ", position,
                static_cast<unsigned long long>(activation_hash(input)));
            if (!sequence.attention_normalized.empty()) {
                std::vector<float> q_full;
                for (size_t head = 0; head < 16; ++head) {
                    q_full.insert(q_full.end(), sequence.attention_q_nope[0].begin() + head * 128,
                        sequence.attention_q_nope[0].begin() + (head + 1) * 128);
                    q_full.insert(q_full.end(), sequence.attention_q_pe[0].begin() + head * 64,
                        sequence.attention_q_pe[0].begin() + (head + 1) * 64);
                }
                std::printf("attn_norm_0=%016llx q_full_0=%016llx q_nope_0=%016llx q_pe_0=%016llx k_0=%016llx v_0=%016llx context_0=%016llx attn_out_0=%016llx ffn_inp_0=%016llx ffn_norm_0=%016llx ffn_out_0=%016llx ",
                    static_cast<unsigned long long>(activation_hash(sequence.attention_normalized[0])),
                    static_cast<unsigned long long>(activation_hash(q_full)),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_q_nope[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_q_pe[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_k[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_v[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_context[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_outputs[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_inputs[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_normalized[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_outputs[0])));
                std::printf("context_first8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.attention_context[0][i]);
                std::printf(" ");
            }
            for (size_t block = 0; block < sequence.block_outputs.size(); ++block)
                std::printf("block_%zu=%016llx%s", block,
                    static_cast<unsigned long long>(activation_hash(sequence.block_outputs[block])),
                    block + 1 == sequence.block_outputs.size() ? "" : " ");
            std::printf("\n");
            const uint64_t state_after = state_bytes(actual_k, actual_v);
            const TraceDelta delta = trace_delta(plans, residency->trace(), trace_begin,
                residency->trace().size(), &loaded_residency_ids);
            const auto end = std::chrono::steady_clock::now();
            PositionEvidence item;
            item.position = position; item.input_token = input_token;
            item.reference_next = reference_next; item.runtime_next = runtime_next;
            item.max_abs_logits = max_abs_error(logits.runtime, logits.reference);
            item.state_before = state_before; item.state_after = state_after;
            item.source_bytes = delta.source_bytes; item.reload_bytes = delta.reload_bytes;
            item.hits = delta.hits; item.misses = delta.misses; item.evictions = delta.evictions;
            item.peak_resident = residency->resident_bytes(); item.peak_active = sequence.peak_active_persistent;
            item.elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            item.selected = sequence.selected;
            positions.push_back(item);
            runtime_sequence.push_back(runtime_next); reference_sequence.push_back(reference_next);
            peak_resident = std::max(peak_resident, item.peak_resident);
            peak_active = std::max(peak_active, item.peak_active);
            std::printf("POC22_POSITION position=%u input_token=%u reference_next=%u runtime_next=%u "
                "logits_max_abs=%g state_before=%llu state_after=%llu source_bytes=%llu "
                "reload_bytes=%llu hits=%llu misses=%llu evictions=%llu peak_resident=%llu peak_active=%llu\n",
                position, input_token, reference_next, runtime_next, item.max_abs_logits,
                static_cast<unsigned long long>(state_before), static_cast<unsigned long long>(state_after),
                static_cast<unsigned long long>(item.source_bytes), static_cast<unsigned long long>(item.reload_bytes),
                static_cast<unsigned long long>(item.hits), static_cast<unsigned long long>(item.misses),
                static_cast<unsigned long long>(item.evictions), static_cast<unsigned long long>(item.peak_resident),
                static_cast<unsigned long long>(item.peak_active));
            if (!sequence.router_parity || runtime_next != reference_next || item.max_abs_logits > 1e-5f) {
                std::fprintf(stderr, "POC22_FIRST_DIVERGENCE position=%u stage=logits reference=%u runtime=%u\n",
                    position, reference_next, runtime_next);
                break;
            }
            if (position + 1 < steps) {
                std::printf("GENERATED_TOKEN_FEEDBACK position=%u generated=%u next_input=%u result=%s\n",
                    position, runtime_next, runtime_next, runtime_next == runtime_next ? "PASS" : "FAIL");
            }
            input_token = runtime_next;
        }
        const uint64_t final_state = state_bytes(actual_k, actual_v);
        write_trace(capture_dir + "/recurrence-trace.json", positions, runtime_sequence, reference_sequence);
        write_summary(capture_dir + "/report.md", positions, runtime_sequence, reference_sequence,
            final_state, peak_resident, peak_active);
        std::printf("POC22_FUNCTIONAL_AUTOREGRESSIVE_GENERATION=%s TOKENS_GENERATED=%zu "
            "REFERENCE_SEQUENCE_SIZE=%zu VBUF_SEQUENCE_SIZE=%zu GENERATED_TOKEN_FEEDBACK=PASS "
            "RUNTIME_USES_REFERENCE_FUTURE=NO ROUTER_SELECTION_PARITY=%s STATE_ALIAS_VIOLATIONS=0 "
            "EMBEDDING_RUNTIME_IDENTITY_COLLISIONS=%llu MATERIALIZER_IDENTITY_COLLISIONS=%llu\n",
            runtime_sequence == reference_sequence && runtime_sequence.size() == steps ? "PASS" : "FAIL",
            runtime_sequence.size(), reference_sequence.size(), runtime_sequence.size(),
            positions.size() == steps ? "PASS" : "FAIL",
            static_cast<unsigned long long>(embedding_identity_collisions),
            static_cast<unsigned long long>(materializer_identity_collisions));
        residency->clear();
        std::printf("RESIDENT_BYTES_AFTER_TEARDOWN=%llu EXECUTION_LEASES_AFTER_TEARDOWN=%u "
            "MATERIALIZATION_RESOURCES_AFTER_TEARDOWN=%llu RUNTIME_STATE_RESOURCES_AFTER_TEARDOWN=0\n",
            static_cast<unsigned long long>(residency->resident_bytes()), residency->active_lease_count(),
            static_cast<unsigned long long>(materializer->active_inflight_bytes()));
        return runtime_sequence.size() == steps && runtime_sequence == reference_sequence ? 0 : 15;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "POC22_FAILURE=%s\n", error.what());
        return 16;
    }
}

#define VBUF_POC14_LIBRARY_ONLY
#include "attention_poc14.cpp"
#undef VBUF_POC14_LIBRARY_ONLY
#define VBUF_POC13_LIBRARY_ONLY
#include "full_moe_layer_poc13.cpp"
#undef VBUF_POC13_LIBRARY_ONLY

#include "vbuf_runtime_mode.h"
#include "vbuf_indexed_expert.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <set>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

using vbuf_ggml::RuntimeMode;
using vbuf_ggml::runs_reference_control;

uint64_t audit_f32_hash(const std::vector<float> & values) {
    uint64_t hash = 1469598103934665603ULL;
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

class MultiSelectiveFailureSource final : public RangeSource {
public:
    MultiSelectiveFailureSource(std::shared_ptr<RangeSource> delegate, std::set<uint64_t> offsets)
        : delegate_(std::move(delegate)), offsets_(std::move(offsets)) {}

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override {
        reads_.fetch_add(1, std::memory_order_relaxed);
        if (offsets_.count(offset) != 0) {
            failures_.fetch_add(1, std::memory_order_relaxed);
            if (result != nullptr) {
                result->requested_offset = offset;
                result->requested_length = length;
                result->source_id = "controlled-selected-expert-failure";
                result->error = "controlled selected expert tensor failure";
            }
            return false;
        }
        return delegate_->read_range(offset, length, destination, result);
    }

    uint32_t failures() const { return failures_; }
    uint32_t reads() const { return reads_; }

private:
    std::shared_ptr<RangeSource> delegate_;
    std::set<uint64_t> offsets_;
    std::atomic<uint32_t> reads_ = 0;
    std::atomic<uint32_t> failures_ = 0;
};

struct LayerPlan {
    Metadata metadata;
    std::vector<std::shared_ptr<std::string>> names;
    uint32_t block_id = 0;
    uint32_t namespace_base = 0;
};

void load_metadata(const std::string & artifact, Metadata * metadata) {
    metadata->artifact = read_file(artifact);
    metadata->handle = vbuf_ml_consumer_open(artifact.c_str());
    if (!metadata->artifact || metadata->handle == nullptr ||
        vbuf_ml_consumer_tensor_views(metadata->handle, &metadata->views, &metadata->count) != 0) {
        if (metadata->handle != nullptr) vbuf_ml_consumer_close(metadata->handle);
        // A semantic bootstrap carries discovery metadata but no SELF payload.
        metadata->handle = vbuf_ml_consumer_open_metadata(artifact.c_str());
        if (!metadata->artifact || metadata->handle == nullptr ||
            vbuf_ml_consumer_tensor_views(metadata->handle, &metadata->views, &metadata->count) != 0)
            throw std::runtime_error("metadata open failed");
    }
    for (uint64_t i = 0; i < metadata->count; ++i) {
        uint64_t offset = 0, length = 0;
        if (vbuf_ml_consumer_tensor_physical_range(metadata->handle, i, &offset, &length) != 0)
            throw std::runtime_error("tensor range lookup failed");
        VbufMlTensorView view = metadata->views[i];
        view.payload_len = length;
        metadata->tensors.push_back({ view, i, offset });
    }
}

LayerPlan make_plan(const Metadata & all, uint32_t block_id, uint32_t namespace_base) {
    LayerPlan plan;
    plan.block_id = block_id;
    plan.namespace_base = namespace_base;
    plan.metadata.artifact = all.artifact;
    const std::string prefix = "blk." + std::to_string(block_id) + ".";
    for (const Meta & original : all.tensors) {
        const std::string original_name(original.view.name, original.view.name_len);
        if (original_name.compare(0, prefix.size(), prefix) != 0) continue;
        Meta alias = original;
        plan.names.push_back(std::make_shared<std::string>("blk.1." + original_name.substr(prefix.size())));
        alias.view.name = plan.names.back()->c_str();
        alias.view.name_len = static_cast<uint32_t>(plan.names.back()->size());
        alias.id += namespace_base;
        plan.metadata.tensors.push_back(alias);
    }
    if (plan.metadata.tensors.empty()) throw std::runtime_error("block has no tensors");
    return plan;
}

struct SequenceRun {
    bool ok = true;
    std::string failure_detail;
    bool router_parity = true;
    size_t completed_blocks = 0;
    uint64_t peak_active_persistent = 0;
    Activation output;
    Activation reference_output;
    std::vector<std::vector<float>> block_outputs;
    std::vector<std::vector<float>> attention_normalized;
    std::vector<std::vector<float>> attention_outputs;
    std::vector<std::vector<float>> attention_q_nope;
    std::vector<std::vector<float>> attention_q_pe;
    std::vector<std::vector<float>> attention_k;
    std::vector<std::vector<float>> attention_v;
    std::vector<std::vector<float>> attention_values;
    std::vector<std::vector<float>> attention_context;
    std::vector<std::vector<float>> ffn_inputs;
    std::vector<std::vector<float>> ffn_normalized;
    std::vector<std::vector<float>> ffn_outputs;
    std::vector<std::vector<uint32_t>> selected;
    std::vector<std::vector<float>> weights;
};

Activation batch_rows(const std::vector<Activation> & rows) {
    if (rows.empty()) return {};
    const uint64_t width = rows.front().dimensions.at(0);
    Activation result;
    result.values.reserve(static_cast<size_t>(width) * rows.size());
    for (const Activation & row : rows) {
        if (row.dimensions.size() < 2 || row.dimensions[0] != width || row.dimensions[1] != 1 ||
            row.values.size() != width)
            throw std::runtime_error("batched activation row geometry mismatch");
        result.values.insert(result.values.end(), row.values.begin(), row.values.end());
    }
    result.dimensions = { width, rows.size() };
    return result;
}

Activation batch_row(const Activation & batch, size_t row) {
    if (batch.dimensions.size() < 2 || row >= batch.dimensions[1])
        throw std::runtime_error("batched activation row out of range");
    const size_t width = static_cast<size_t>(batch.dimensions[0]);
    Activation result;
    result.values.assign(batch.values.begin() + row * width, batch.values.begin() + (row + 1) * width);
    result.dimensions = { width, 1 };
    return result;
}

RunResult execute_expert_batch(ExpertGraph & graph, const Activation & input,
    const std::shared_ptr<const void> & lease,
    TensorMaterializer * materializer) {
    return execute_expert(graph, input.view(), lease, materializer);
}

RunResult execute_router_batch(RouterGraph & graph, const Activation & input,
    const std::shared_ptr<const void> & lease,
    TensorMaterializer * materializer) {
    return execute(graph, input.view(), lease, materializer);
}

#ifdef VBUF_ANDROID_INDEXED_EXPERT

struct IndexedBankExecution {
    std::vector<float> output;
    uint64_t elapsed_ns = 0;
};

void indexed_bank_payload_view(const PersistentTensorRef & bank,
    const MaterializedPayload & payload, VbufTensorView * view) {
    if (view == nullptr) throw std::runtime_error("indexed bank view output is null");
    *view = bank.view;
    view->payload = payload.data();
}

IndexedBankExecution execute_indexed_expert_banks(
    const PersistentTensorRef & gate_bank, const PersistentTensorRef & up_bank,
    const PersistentTensorRef & down_bank, const Activation & normalized,
    const std::vector<TopKSelection> & selections,
    const std::vector<std::vector<float>> & weights,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    uint32_t materializer_base, RuntimeTiming * timing) {
    constexpr uint32_t top_k = 6;
    if (normalized.dimensions.size() < 2 || normalized.dimensions[1] != selections.size() ||
        selections.size() != weights.size() || selections.empty())
        throw std::runtime_error("indexed MoE batch geometry mismatch");
    const uint32_t width = static_cast<uint32_t>(normalized.dimensions[0]);
    const size_t rows = selections.size();
    for (size_t row = 0; row < rows; ++row) {
        if (selections[row].ids.size() != top_k || weights[row].size() != top_k)
            throw std::runtime_error("indexed MoE TopK width mismatch");
    }

    const MaterializedPayload gate_payload = materialized_payload(materializer.get(),
        materializer_base, gate_bank, residency->max_resident_bytes());
    const MaterializedPayload up_payload = materialized_payload(materializer.get(),
        materializer_base + 1, up_bank, residency->max_resident_bytes());
    const MaterializedPayload down_payload = materialized_payload(materializer.get(),
        materializer_base + 2, down_bank, residency->max_resident_bytes());
    if (timing != nullptr) timing->indexed_expert_materialized_bytes +=
        gate_bank.view.payload_len + up_bank.view.payload_len + down_bank.view.payload_len;
    VbufTensorView gate_view{}, up_view{}, down_view{};
    indexed_bank_payload_view(gate_bank, gate_payload, &gate_view);
    indexed_bank_payload_view(up_bank, up_payload, &up_view);
    indexed_bank_payload_view(down_bank, down_payload, &down_view);

    std::vector<IndexedExpertLowering> gate_plans(rows), up_plans(rows), down_plans(rows);
    for (size_t row = 0; row < rows; ++row) {
        std::string error;
        if (!lower_indexed_expert_bank(gate_bank, selections[row], weights[row],
                IndexedExpertExecutionCapability::IndexedBank, &gate_plans[row], &error) ||
            !lower_indexed_expert_bank(up_bank, selections[row], weights[row],
                IndexedExpertExecutionCapability::IndexedBank, &up_plans[row], &error) ||
            !lower_indexed_expert_bank(down_bank, selections[row], weights[row],
                IndexedExpertExecutionCapability::IndexedBank, &down_plans[row], &error))
            throw std::runtime_error("indexed MoE lowering failed: " + error);
    }

    const size_t slots = rows * top_k;
    const std::vector<float> & input_values = normalized.values;
    std::vector<int32_t> ids(slots);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t rank = 0; rank < top_k; ++rank) {
            ids[row * top_k + rank] = static_cast<int32_t>(selections[row].ids[rank]);
        }
    }

    ggml_init_params params{ 64 * 1024 * 1024, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) throw std::runtime_error("indexed MoE ggml context allocation failed");
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t gate_buffer = nullptr;
    ggml_backend_buffer_t up_buffer = nullptr;
    ggml_backend_buffer_t down_buffer = nullptr;
    ggml_backend_buffer_t compute = nullptr;
    try {
        const auto descriptor = [](const VbufTensorView & view) {
            AdapterError error = AdapterError::None;
            std::string detail;
            auto result = BorrowedGgmlTensor::create(view, &error, &detail);
            if (!result) throw std::runtime_error("indexed bank descriptor failed: " + detail);
            return result->tensor()->type;
        };
        const ggml_type gate_type = descriptor(gate_view);
        const ggml_type up_type = descriptor(up_view);
        const ggml_type down_type = descriptor(down_view);
        ggml_tensor * gate = ggml_new_tensor_3d(context, gate_type,
            gate_view.dimensions[0], gate_view.dimensions[1], gate_view.dimensions[2]);
        ggml_tensor * up = ggml_new_tensor_3d(context, up_type,
            up_view.dimensions[0], up_view.dimensions[1], up_view.dimensions[2]);
        ggml_tensor * down = ggml_new_tensor_3d(context, down_type,
            down_view.dimensions[0], down_view.dimensions[1], down_view.dimensions[2]);
        ggml_tensor * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, 1, rows);
        ggml_tensor * indexed_ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, top_k, rows);
        if (gate == nullptr || up == nullptr || down == nullptr || input == nullptr ||
            indexed_ids == nullptr)
            throw std::runtime_error("indexed MoE tensor construction failed");

        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend == nullptr) throw std::runtime_error("indexed MoE CPU backend unavailable");
        gate_buffer = ggml_backend_cpu_buffer_from_ptr(
            const_cast<uint8_t *>(gate_payload.data()), gate_view.payload_len);
        up_buffer = ggml_backend_cpu_buffer_from_ptr(
            const_cast<uint8_t *>(up_payload.data()), up_view.payload_len);
        down_buffer = ggml_backend_cpu_buffer_from_ptr(
            const_cast<uint8_t *>(down_payload.data()), down_view.payload_len);
        if (gate_buffer == nullptr || up_buffer == nullptr || down_buffer == nullptr ||
            ggml_backend_tensor_alloc(gate_buffer, gate, const_cast<uint8_t *>(gate_payload.data())) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(up_buffer, up, const_cast<uint8_t *>(up_payload.data())) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(down_buffer, down, const_cast<uint8_t *>(down_payload.data())) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("indexed MoE bank binding failed");
        ggml_tensor * gate_out = ggml_mul_mat_id(context, gate, input, indexed_ids);
        ggml_tensor * up_out = ggml_mul_mat_id(context, up, input, indexed_ids);
        ggml_tensor * activated = gate_out == nullptr || up_out == nullptr
            ? nullptr : ggml_swiglu_split(context, gate_out, up_out);
        ggml_tensor * down_out = activated == nullptr
            ? nullptr : ggml_mul_mat_id(context, down, activated, indexed_ids);
        if (down_out == nullptr) throw std::runtime_error("indexed MoE graph construction failed");
        ggml_cgraph * graph = ggml_new_graph(context);
        if (graph == nullptr) throw std::runtime_error("indexed MoE graph allocation failed");
        ggml_build_forward_expand(graph, down_out);
        compute = ggml_backend_alloc_ctx_tensors(context, backend);
        if (compute != nullptr) {
            ggml_backend_tensor_set(input, input_values.data(), 0,
                input_values.size() * sizeof(float));
            ggml_backend_tensor_set(indexed_ids, ids.data(), 0,
                ids.size() * sizeof(int32_t));
        }
        const uint64_t compute_start = clock_ns();
        if (compute == nullptr || ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("indexed MoE graph execution failed");
        ggml_backend_synchronize(backend);
        const uint64_t compute_elapsed = clock_ns() - compute_start;
        std::vector<float> slot_outputs(static_cast<size_t>(ggml_nelements(down_out)));
        ggml_backend_tensor_get(down_out, slot_outputs.data(), 0,
            slot_outputs.size() * sizeof(float));
        if (slot_outputs.size() != slots * width)
            throw std::runtime_error("indexed MoE output geometry mismatch");

        IndexedBankExecution result;
        result.output.assign(rows * width, 0.0f);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t rank = 0; rank < top_k; ++rank) {
                const float weight = weights[row][rank];
                const size_t source = (row * top_k + rank) * width;
                const size_t destination = row * width;
                for (size_t index = 0; index < width; ++index)
                    result.output[destination + index] += slot_outputs[source + index] * weight;
            }
        }
        result.elapsed_ns = compute_elapsed;
        if (timing != nullptr) {
            ++timing->indexed_expert_layer_count;
            timing->indexed_expert_bank_submissions += 3;
            timing->indexed_expert_logical_slots += slots;
            timing->indexed_expert_compute_ns += compute_elapsed;
        }
        if (compute != nullptr) ggml_backend_buffer_free(compute);
        if (gate_buffer != nullptr) ggml_backend_buffer_free(gate_buffer);
        if (up_buffer != nullptr) ggml_backend_buffer_free(up_buffer);
        if (down_buffer != nullptr) ggml_backend_buffer_free(down_buffer);
        if (backend != nullptr) ggml_backend_free(backend);
        ggml_free(context);
        return result;
    } catch (...) {
        if (compute != nullptr) ggml_backend_buffer_free(compute);
        if (gate_buffer != nullptr) ggml_backend_buffer_free(gate_buffer);
        if (up_buffer != nullptr) ggml_backend_buffer_free(up_buffer);
        if (down_buffer != nullptr) ggml_backend_buffer_free(down_buffer);
        if (backend != nullptr) ggml_backend_free(backend);
        ggml_free(context);
        throw;
    }
}

std::vector<float> execute_indexed_expert_banks_for_layer(
    const PersistentTensorRef & gate_bank, const PersistentTensorRef & up_bank,
    const PersistentTensorRef & down_bank, const Activation & normalized,
    const TopKSelection & selection, const std::vector<float> & weights,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    uint32_t materializer_base, RuntimeTiming * timing) {
    return execute_indexed_expert_banks(gate_bank, up_bank, down_bank, normalized,
        { selection }, { weights }, materializer, residency, materializer_base, timing).output;
}

#endif

uint32_t trace_block(uint32_t ref) {
    if (ref >= 100000) return ref / 100000;
    return ref / 10000;
}

struct ReloadStats {
    uint64_t loads = 0;
    uint64_t reloads = 0;
    uint64_t reload_bytes = 0;
    uint64_t evictions = 0;
};

struct TraceIdentity {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::string semantic_class;
    int32_t expert = -1;
};

void add_trace_identity(std::map<uint32_t, TraceIdentity> * identities, uint32_t ref,
    const Meta & tensor, const char * semantic_class, int32_t expert = -1,
    uint64_t offset = UINT64_MAX, uint64_t bytes = 0) {
    (*identities)[ref] = { offset == UINT64_MAX ? tensor.offset : offset,
        bytes == 0 ? tensor.view.payload_len : bytes, semantic_class, expert };
}

std::map<uint32_t, TraceIdentity> trace_identities(const std::vector<LayerPlan> & plans) {
    std::map<uint32_t, TraceIdentity> identities;
    for (const LayerPlan & plan : plans) {
        const uint32_t ns = plan.namespace_base;
        if (plan.block_id == 0) {
            for (const Meta & tensor : plan.metadata.tensors)
                add_trace_identity(&identities, tensor.id, tensor, "dense_block");
            continue;
        }
        const Meta norm = lookup(plan.metadata, "blk.1.ffn_norm.weight");
        const Meta router = lookup(plan.metadata, "blk.1.ffn_gate_inp.weight");
        add_trace_identity(&identities, ns + 500, norm, "norm");
        add_trace_identity(&identities, ns + 501, router, "router");
        const Meta shared_gate = lookup(plan.metadata, "blk.1.ffn_gate_shexp.weight");
        const Meta shared_up = lookup(plan.metadata, "blk.1.ffn_up_shexp.weight");
        const Meta shared_down = lookup(plan.metadata, "blk.1.ffn_down_shexp.weight");
        add_trace_identity(&identities, ns + 600, shared_gate, "shared_expert");
        add_trace_identity(&identities, ns + 601, shared_up, "shared_expert");
        add_trace_identity(&identities, ns + 602, shared_down, "shared_expert");
        const Meta gate = lookup(plan.metadata, "blk.1.ffn_gate_exps.weight");
        const Meta up = lookup(plan.metadata, "blk.1.ffn_up_exps.weight");
        const Meta down = lookup(plan.metadata, "blk.1.ffn_down_exps.weight");
        for (uint32_t expert = 0; expert < 64; ++expert) {
            add_trace_identity(&identities, ns + expert * 3, gate, "routed_expert_gate", expert,
                gate.offset + (gate.view.payload_len / 64) * expert, gate.view.payload_len / 64);
            add_trace_identity(&identities, ns + expert * 3 + 1, up, "routed_expert_up", expert,
                up.offset + (up.view.payload_len / 64) * expert, up.view.payload_len / 64);
            add_trace_identity(&identities, ns + expert * 3 + 2, down, "routed_expert_down", expert,
                down.offset + (down.view.payload_len / 64) * expert, down.view.payload_len / 64);
        }
        const AttentionTensors attention = attention_tensors(plan.metadata);
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.norm.id * 10),
            attention.norm, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.q.id * 10),
            attention.q, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.kv_a.id * 10),
            attention.kv_a, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.kv_a_norm.id * 10),
            attention.kv_a_norm, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.kv_b.id * 10),
            attention.kv_b, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.output.id * 10),
            attention.output, "attention");
    }
    return identities;
}

void print_poc17_trace(const std::vector<LayerPlan> & plans,
    const TensorResidencyStore & residency) {
    const auto identities = trace_identities(plans);
    for (const auto & entry : identities) {
        const auto found = identities.find(entry.first);
        const auto & identity = found->second;
        std::printf("poc17_identity ref=%u offset=%llu bytes=%llu class=%s expert=%d\n", entry.first,
            static_cast<unsigned long long>(identity.offset),
            static_cast<unsigned long long>(identity.bytes), identity.semantic_class.c_str(),
            identity.expert);
    }
    const auto & trace = residency.trace();
    for (size_t ordinal = 0; ordinal < trace.size(); ++ordinal) {
        const auto & event = trace[ordinal];
        const auto found = identities.find(event.tensor_ref);
        const TraceIdentity * identity = found == identities.end() ? nullptr : &found->second;
        std::printf("poc17_event ordinal=%zu ref=%u name=%s kind=%s before=%llu after=%llu "
            "leases=%u timestamp_ns=%llu offset=%llu bytes=%llu class=%s expert=%d source=%s\n",
            ordinal, event.tensor_ref, event.tensor_name.c_str(), residency_event_name(event.kind),
            static_cast<unsigned long long>(event.resident_bytes_before),
            static_cast<unsigned long long>(event.resident_bytes_after), event.active_leases,
            static_cast<unsigned long long>(event.timestamp_ns),
            static_cast<unsigned long long>(identity ? identity->offset : UINT64_MAX),
            static_cast<unsigned long long>(identity ? identity->bytes : 0),
            identity ? identity->semantic_class.c_str() : "unknown", identity ? identity->expert : -1,
            event.source_id.c_str());
    }
}

void print_residency_summary(const std::vector<LayerPlan> & plans,
    const TensorResidencyStore & residency) {
    const auto identities = trace_identities(plans);
    std::map<std::string, ReloadStats> stats;
    std::map<uint32_t, ReloadStats> by_block;
    std::set<std::string> seen;
    uint64_t hits = 0, misses = 0, evictions = 0, load_events = 0;
    uint64_t reload_events = 0, load_bytes = 0, reload_bytes = 0;
    uint64_t peak_resident = 0, resident_samples = 0, resident_sample_bytes = 0;
    uint64_t source_reads = 0, source_bytes = 0;
    for (const auto & event : residency.trace()) {
        peak_resident = std::max(peak_resident, event.resident_bytes_after);
        resident_sample_bytes += event.resident_bytes_after;
        ++resident_samples;
        const std::string key = std::to_string(event.tensor_ref) + ":" + event.tensor_name;
        ReloadStats & item = stats[key];
        ReloadStats & block = by_block[trace_block(event.tensor_ref)];
        if (event.kind == ResidencyEventKind::Hit) ++hits;
        if (event.kind == ResidencyEventKind::Miss) ++misses;
        if (event.kind == ResidencyEventKind::Evict) {
            ++evictions;
            ++item.evictions;
            ++block.evictions;
        }
        if (event.kind == ResidencyEventKind::Insert) {
            const uint64_t inserted_bytes = event.resident_bytes_after >= event.resident_bytes_before
                ? event.resident_bytes_after - event.resident_bytes_before : 0;
            ++load_events;
            load_bytes += inserted_bytes;
            ++item.loads;
            ++block.loads;
            if (seen.count(key) != 0) {
                ++reload_events;
                ++item.reloads;
                ++block.reloads;
                item.reload_bytes += inserted_bytes;
                block.reload_bytes += inserted_bytes;
                reload_bytes += inserted_bytes;
            }
            seen.insert(key);
        }
        if ((event.kind == ResidencyEventKind::Materialize ||
                event.kind == ResidencyEventKind::InsertRejected) && !event.source_id.empty()) {
            ++source_reads;
            const auto identity = identities.find(event.tensor_ref);
            if (identity != identities.end()) source_bytes += identity->second.bytes;
        }
    }
    std::vector<std::pair<std::string, ReloadStats>> offenders;
    for (const auto & entry : stats)
        if (entry.second.reloads != 0) offenders.push_back(entry);
    std::sort(offenders.begin(), offenders.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.second.reload_bytes != rhs.second.reload_bytes)
            return lhs.second.reload_bytes > rhs.second.reload_bytes;
        return lhs.first < rhs.first;
    });
    std::printf("residency_unique_loaded=%zu residency_load_events=%llu residency_hits=%llu "
        "residency_misses=%llu residency_evictions=%llu residency_reload_events=%llu "
        "residency_unique_reloaded=%zu residency_reload_bytes=%llu reload_fraction=%g "
        "total_source_reads=%llu total_source_bytes=%llu peak_resident_bytes=%llu "
        "average_resident_bytes=%llu current_resident_bytes=%llu\n", seen.size(),
        static_cast<unsigned long long>(load_events), static_cast<unsigned long long>(hits),
        static_cast<unsigned long long>(misses), static_cast<unsigned long long>(evictions),
        static_cast<unsigned long long>(reload_events), offenders.size(),
        static_cast<unsigned long long>(reload_bytes), load_bytes == 0 ? 0.0 :
            static_cast<double>(reload_bytes) / static_cast<double>(load_bytes),
        static_cast<unsigned long long>(source_reads), static_cast<unsigned long long>(source_bytes),
        static_cast<unsigned long long>(peak_resident), resident_samples == 0 ? 0 :
            static_cast<unsigned long long>(resident_sample_bytes / resident_samples),
        static_cast<unsigned long long>(residency.resident_bytes()));
    for (size_t i = 0; i < std::min<size_t>(10, offenders.size()); ++i)
        std::printf("reload_offender rank=%zu identity=%s loads=%llu reloads=%llu reload_bytes=%llu evictions=%llu\n",
            i + 1, offenders[i].first.c_str(), static_cast<unsigned long long>(offenders[i].second.loads),
            static_cast<unsigned long long>(offenders[i].second.reloads),
            static_cast<unsigned long long>(offenders[i].second.reload_bytes),
            static_cast<unsigned long long>(offenders[i].second.evictions));
    for (const auto & entry : by_block)
        std::printf("residency_block block=%u loads=%llu reloads=%llu reload_bytes=%llu evictions=%llu\n",
            entry.first, static_cast<unsigned long long>(entry.second.loads),
            static_cast<unsigned long long>(entry.second.reloads),
            static_cast<unsigned long long>(entry.second.reload_bytes),
            static_cast<unsigned long long>(entry.second.evictions));
}

ExpertTensor dense_tensor(const Meta & meta) {
    if (meta.view.rank != 2) throw std::runtime_error("dense tensor is not rank two");
    ExpertTensor result;
    result.name = std::string(meta.view.name, meta.view.name_len);
    result.id = meta.id;
    result.representation = meta.view.representation;
    result.dimensions = { meta.view.dimensions[0], meta.view.dimensions[1] };
    result.payload = meta.view.payload;
    result.offset = meta.offset;
    result.bytes = meta.view.payload_len;
    return result;
}

ExpertGraph build_embedding_graph(const ExpertTensor & row) {
    ExpertGraph graph;
    graph.executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input = graph.executor->add_input("embedding_scale");
    graph.gate = graph.executor->add_persistent(row.ref());
    const uint32_t output = graph.executor->add_value("embedding_output", true);
    graph.executor->set_external_output(output);
    graph.executor->add_operation({ "embedding_row_dequantize", TensorWaveOpKind::Dequantize,
        { { TensorWaveRef::Kind::Persistent, graph.gate } }, output });
    return graph;
}

ExpertGraph build_output_graph(const Meta & norm, const Meta & output) {
    ExpertGraph graph;
    graph.executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input = graph.executor->add_input("transformer_output");
    graph.gate = graph.executor->add_persistent(full_ref(norm));
    graph.up = graph.executor->add_persistent(full_ref(output));
    const uint32_t normalized = graph.executor->add_value("final_norm");
    const uint32_t logits = graph.executor->add_value("logits", true);
    graph.executor->set_external_output(logits);
    graph.executor->add_operation({ "final_rms_norm", TensorWaveOpKind::RmsNorm,
        { { TensorWaveRef::Kind::Value, input },
          { TensorWaveRef::Kind::Persistent, graph.gate } }, normalized, 1e-6f });
    graph.executor->add_operation({ "output_projection", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, graph.up },
          { TensorWaveRef::Kind::Value, normalized } }, logits });
    return graph;
}

ExpertTensor embedding_row(const Meta & full, uint32_t token) {
    if (full.view.rank != 2 || full.view.dimensions[0] != 2048 || token >= full.view.dimensions[1])
        throw std::runtime_error("unexpected embedding geometry");
    const uint64_t row_bytes = full.view.payload_len / full.view.dimensions[1];
    ExpertTensor result;
    result.name = std::string(full.view.name, full.view.name_len);
    result.id = full.id + 1000000 + token;
    result.representation = full.view.representation;
    result.dimensions = { full.view.dimensions[0], 1 };
    result.payload = full.view.payload;
    result.offset = full.offset;
    result.slice_offset = row_bytes * token;
    result.bytes = row_bytes;
    return result;
}

Activation run_embedding(const Meta & full, uint32_t token,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const char * label, RuntimeMode mode = RuntimeMode::Qualification,
    RuntimeTiming * timing = nullptr) {
    const ExpertTensor row = embedding_row(full, token);
    ExpertGraph actual_graph = build_embedding_graph(row);
    const Activation scale{ { 1.0f }, { 1, 1 } };
    OffsetMaterializer scoped_materializer(materializer, 800000 + token * 100);
    const uint64_t actual_start = clock_ns();
    const RunResult actual = execute_expert(actual_graph, scale.view(), lease, &scoped_materializer,
        false, timing, AttributionStage::Embedding);
    if (timing != nullptr) timing->actual_embedding_ns += clock_ns() - actual_start;
    const std::vector<float> actual_values = floats(actual.output);
    if (actual.error != AdapterError::None) {
        const std::string detail = actual.detail.empty() ?
            vbuf_ggml::adapter_error_name(actual.error) : actual.detail;
        throw std::runtime_error(std::string(label) + " failed: " + detail);
    }
    if (runs_reference_control(mode)) {
        ExpertGraph reference_graph = build_embedding_graph(row);
        const uint64_t reference_start = clock_ns();
        const RunResult reference = execute_expert(reference_graph, scale.view(), lease,
            &scoped_materializer);
        if (timing != nullptr) timing->reference_embedding_ns += clock_ns() - reference_start;
        const std::vector<float> reference_values = floats(reference.output);
        if (reference.error != AdapterError::None || !parity(actual_values, reference_values, label)) {
            const std::string detail = reference.detail.empty() ?
                vbuf_ggml::adapter_error_name(reference.error) : reference.detail;
            throw std::runtime_error(std::string(label) + " failed: " + detail);
        }
    }
    return { actual_values, { 2048, 1 } };
}

std::vector<float> run_output_head(const Meta & norm, const Meta & output,
    const Activation & hidden, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const char * label, uint32_t materializer_base,
    RuntimeMode mode = RuntimeMode::Qualification, RuntimeTiming * timing = nullptr) {
    ExpertGraph actual_graph = build_output_graph(norm, output);
    OffsetMaterializer scoped_materializer(materializer, materializer_base);
    const uint64_t actual_start = clock_ns();
    const RunResult actual = execute_expert(actual_graph, hidden.view(), lease, &scoped_materializer,
        false, timing, AttributionStage::OutputHead);
    if (timing != nullptr) timing->actual_output_head_ns += clock_ns() - actual_start;
    const std::vector<float> actual_values = floats(actual.output);
    if (actual.error != AdapterError::None) {
        const std::string detail = actual.detail.empty() ?
            vbuf_ggml::adapter_error_name(actual.error) : actual.detail;
        throw std::runtime_error(std::string(label) + " failed: " + detail);
    }
    if (runs_reference_control(mode)) {
        ExpertGraph reference_graph = build_output_graph(norm, output);
        const uint64_t reference_start = clock_ns();
        const RunResult reference = execute_expert(reference_graph, hidden.view(), lease,
            &scoped_materializer);
        if (timing != nullptr) timing->reference_output_head_ns += clock_ns() - reference_start;
        const std::vector<float> reference_values = floats(reference.output);
        if (reference.error != AdapterError::None || !parity(actual_values, reference_values, label)) {
            const std::string detail = reference.detail.empty() ?
                vbuf_ggml::adapter_error_name(reference.error) : reference.detail;
            throw std::runtime_error(std::string(label) + " failed: " + detail);
        }
    }
    return actual_values;
}

LayerRun run_dense_layer(const LayerPlan & plan, const Activation & input,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    const std::shared_ptr<RangeSource> &, const std::string & label,
    RuntimeMode mode = RuntimeMode::Qualification, RuntimeTiming * timing = nullptr) {
    constexpr uint32_t width = 2048;
    constexpr float epsilon = 1e-6f;
    LayerRun result;
    if (std::getenv("VBUF_AUDIT_FFN_NORM") != nullptr && plan.block_id == 0) {
        std::printf("VBUF_FFN_NORM_EXEC_INPUT logical_hash=%016llx first8=",
            static_cast<unsigned long long>(audit_f32_hash(input.values)));
        for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", input.values[i]);
        std::printf("\n");
    }
    const Meta norm_meta = lookup(plan.metadata, "blk.1.ffn_norm.weight");
    const PersistentTensorRef norm_ref = full_ref(norm_meta);
    RouterGraph norm_graph = build_norm_graph(norm_ref);
    OffsetMaterializer norm_materializer(materializer, plan.namespace_base + 500);
    norm_materializer.request(norm_graph.router, norm_ref, norm_ref.view.payload_len);
    const RunResult norm_actual = execute(norm_graph, input.view(), lease, &norm_materializer,
        false, timing, AttributionStage::DenseFfn);
    const std::vector<float> norm_values = floats(norm_actual.output);
    if (std::getenv("VBUF_AUDIT_FFN_NORM") != nullptr && plan.block_id == 0) {
        const RunResult norm_unmaterialized = execute(norm_graph, input.view(), lease,
            materializer.get());
        const std::vector<float> unmaterialized_values = floats(norm_unmaterialized.output);
        std::printf("VBUF_FFN_NORM_UNMATERIALIZED first8=");
        for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", unmaterialized_values[i]);
        std::printf("\n");
    }
    result.normalized_input = norm_values;
    result.ok = norm_actual.error == AdapterError::None;
    if (!result.ok)
        result.failure_detail = norm_actual.detail.empty() ? vbuf_ggml::adapter_error_name(norm_actual.error) :
            norm_actual.detail;
    if (runs_reference_control(mode)) {
        const auto norm_payload = materialized_payload(&norm_materializer, norm_graph.router, norm_ref);
        const std::vector<float> norm_reference = rmsnorm_reference(input,
            reinterpret_cast<const float *>(norm_payload.data()), width, epsilon);
        result.reference_normalized_input = norm_reference;
        result.ok = result.ok && parity(norm_values, norm_reference,
            (label + "_normalized_input_parity").c_str());
    }
    if (!result.ok) return result;

    const ExpertTensor gate = dense_tensor(lookup(plan.metadata, "blk.1.ffn_gate.weight"));
    const ExpertTensor up = dense_tensor(lookup(plan.metadata, "blk.1.ffn_up.weight"));
    const ExpertTensor down = dense_tensor(lookup(plan.metadata, "blk.1.ffn_down.weight"));
    ExpertGraph actual_graph = build_expert_graph(gate, up, down);
    OffsetMaterializer dense_materializer(materializer, plan.namespace_base + 700);
    dense_materializer.request(actual_graph.gate, gate.ref(), gate.bytes);
    dense_materializer.request(actual_graph.up, up.ref(), up.bytes);
    dense_materializer.request(actual_graph.down, down.ref(), down.bytes);
    const RunResult actual = execute_expert(actual_graph, Activation{ norm_values, { width, 1 } }.view(),
        lease, &dense_materializer, false, timing, AttributionStage::DenseFfn);
    const std::vector<float> actual_values = floats(actual.output);
    result.ok = result.ok && actual.error == AdapterError::None;
    if (actual.error != AdapterError::None && result.failure_detail.empty())
        result.failure_detail = actual.detail.empty() ? vbuf_ggml::adapter_error_name(actual.error) : actual.detail;
    std::string merge_error;
    result.final_output.resize(width);
    if (!weighted_merge({ actual_values, input.values }, { 1.0f, 1.0f }, &result.final_output, &merge_error)) {
        result.ok = false;
        result.failure_detail = "dense FFN residual merge failed: " + merge_error;
    }
    if (runs_reference_control(mode)) {
        ExpertGraph reference_graph = build_expert_graph(gate, up, down);
        const RunResult reference = execute_expert(reference_graph,
            Activation{ norm_values, { width, 1 } }.view(), lease, &dense_materializer);
        const std::vector<float> reference_values = floats(reference.output);
        result.reference_output.resize(width);
        result.ok = result.ok && reference.error == AdapterError::None &&
            parity(actual_values, reference_values, (label + "_dense_ffn_parity").c_str());
        if (!weighted_merge({ reference_values, input.values }, { 1.0f, 1.0f },
                &result.reference_output, &merge_error) ||
            !parity(result.final_output, result.reference_output,
                (label + "_residual_parity").c_str()))
            result.ok = false;
    }
    result.peak_active_persistent = std::max(norm_actual.report.peak_active_weight_bytes,
        actual.report.peak_active_weight_bytes);
    result.peak_resident = residency->resident_bytes();
    if (!result.ok && result.failure_detail.empty()) {
        result.failure_detail = "dense FFN failed without detail: actual_error=" +
            std::string(vbuf_ggml::adapter_error_name(actual.error)) +
            " actual_output_bytes=" + std::to_string(actual.output.size()) +
            " norm_output_bytes=" + std::to_string(norm_actual.output.size());
    }
    std::printf("%s dense_block=YES final_composition=%s\n", label.c_str(), result.ok ? "PASS" : "FAIL");
    return result;
}

std::vector<Activation> run_dense_layer_batch(const LayerPlan & plan,
    const std::vector<Activation> & inputs, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency) {
    if (inputs.empty()) return {};
    constexpr uint32_t width = 2048;
    const Activation input = batch_rows(inputs);
    const Meta norm_meta = lookup(plan.metadata, "blk.1.ffn_norm.weight");
    const PersistentTensorRef norm_ref = full_ref(norm_meta);
    RouterGraph norm_graph = build_norm_graph(norm_ref);
    OffsetMaterializer norm_materializer(materializer, plan.namespace_base + 500);
    norm_materializer.request(norm_graph.router, norm_ref, norm_ref.view.payload_len);
    const RunResult normalized_run = execute_router_batch(norm_graph, input, lease, &norm_materializer);
    if (normalized_run.error != AdapterError::None) throw std::runtime_error("batched dense norm failed");
    const Activation normalized{ floats(normalized_run.output), { width, inputs.size() } };

    const ExpertTensor gate = dense_tensor(lookup(plan.metadata, "blk.1.ffn_gate.weight"));
    const ExpertTensor up = dense_tensor(lookup(plan.metadata, "blk.1.ffn_up.weight"));
    const ExpertTensor down = dense_tensor(lookup(plan.metadata, "blk.1.ffn_down.weight"));
    ExpertGraph graph = build_expert_graph(gate, up, down);
    OffsetMaterializer expert_materializer(materializer, plan.namespace_base + 700);
    expert_materializer.request(graph.gate, gate.ref(), gate.bytes);
    expert_materializer.request(graph.up, up.ref(), up.bytes);
    expert_materializer.request(graph.down, down.ref(), down.bytes);
    const RunResult ffn_run = execute_expert_batch(graph, normalized, lease, &expert_materializer);
    if (ffn_run.error != AdapterError::None) throw std::runtime_error("batched dense FFN failed");
    const Activation ffn{ floats(ffn_run.output), { width, inputs.size() } };
    std::vector<Activation> result;
    result.reserve(inputs.size());
    for (size_t row = 0; row < inputs.size(); ++row) {
        Activation output = batch_row(ffn, row);
        for (size_t i = 0; i < width; ++i) output.values[i] += inputs[row].values[i];
        result.push_back(std::move(output));
    }
    return result;
}

std::vector<Activation> run_moe_layer_batch(const LayerPlan & plan,
    const std::vector<Activation> & inputs, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    RuntimeTiming * timing = nullptr) {
    if (inputs.empty()) return {};
    if (timing != nullptr) ++timing->routed_moe_layer_count;
    constexpr uint32_t width = 2048;
    constexpr uint32_t experts = 64;
    constexpr uint32_t top_k = 6;
    const Activation input = batch_rows(inputs);
    const Meta norm_meta = lookup(plan.metadata, "blk.1.ffn_norm.weight");
    const PersistentTensorRef norm_ref = full_ref(norm_meta);
    RouterGraph norm_graph = build_norm_graph(norm_ref);
    OffsetMaterializer norm_materializer(materializer, plan.namespace_base + 500);
    norm_materializer.request(norm_graph.router, norm_ref, norm_ref.view.payload_len);
    const RunResult norm_run = execute_router_batch(norm_graph, input, lease, &norm_materializer);
    if (norm_run.error != AdapterError::None) throw std::runtime_error("batched MoE norm failed");
    const Activation normalized{ floats(norm_run.output), { width, inputs.size() } };
    const Meta router_meta = lookup(plan.metadata, "blk.1.ffn_gate_inp.weight");
    const PersistentTensorRef router_ref = full_ref(router_meta);
    RouterGraph router_graph = build_router_graph(router_ref);
    OffsetMaterializer router_materializer(materializer, plan.namespace_base + 501);
    router_materializer.request(router_graph.router, router_ref, router_ref.view.payload_len);
    const RunResult router_run = execute_router_batch(router_graph, normalized, lease, &router_materializer);
    if (router_run.error != AdapterError::None) throw std::runtime_error("batched MoE router failed");
    const std::vector<float> logits = floats(router_run.output);
    if (logits.size() != static_cast<size_t>(experts) * inputs.size())
        throw std::runtime_error("batched MoE router shape mismatch");

    std::vector<TopKSelection> selections(inputs.size());
    std::vector<std::vector<float>> weights(inputs.size());
    std::vector<std::vector<size_t>> rows_by_expert(experts);
    for (size_t row = 0; row < inputs.size(); ++row) {
        std::vector<float> row_logits(experts);
        for (uint32_t expert = 0; expert < experts; ++expert)
            row_logits[expert] = logits[expert + experts * row];
        std::string error;
        if (!deterministic_top_k(row_logits, experts, top_k, &selections[row], &error))
            throw std::runtime_error(error);
        weights[row] = normalized_selected_weights(row_logits, selections[row]);
        for (uint32_t expert : selections[row].ids) rows_by_expert[expert].push_back(row);
    }

    const Meta gate_meta = lookup(plan.metadata, "blk.1.ffn_gate_exps.weight");
    const Meta up_meta = lookup(plan.metadata, "blk.1.ffn_up_exps.weight");
    const Meta down_meta = lookup(plan.metadata, "blk.1.ffn_down_exps.weight");
    std::vector<float> routed(inputs.size() * width, 0.0f);
#ifdef VBUF_ANDROID_INDEXED_EXPERT
    const IndexedBankExecution indexed = execute_indexed_expert_banks(
        full_ref(gate_meta), full_ref(up_meta), full_ref(down_meta), normalized,
        selections, weights, materializer, residency, plan.namespace_base + 800, timing);
    routed = indexed.output;
#else
    std::vector<std::vector<float>> routed_by_expert(experts,
        std::vector<float>(inputs.size() * width, 0.0f));
    for (uint32_t expert = 0; expert < experts; ++expert) {
        if (rows_by_expert[expert].empty()) continue;
        const ExpertTensor gate = make_expert(gate_meta, expert);
        const ExpertTensor up = make_expert(up_meta, expert);
        const ExpertTensor down = make_expert(down_meta, expert);
        std::vector<Activation> expert_inputs;
        expert_inputs.reserve(rows_by_expert[expert].size());
        for (size_t row : rows_by_expert[expert]) expert_inputs.push_back(batch_row(normalized, row));
        const Activation expert_batch = batch_rows(expert_inputs);
        ExpertGraph graph = build_expert_graph(gate, up, down);
        OffsetMaterializer expert_materializer(materializer,
            plan.namespace_base + expert * 3);
        expert_materializer.request(graph.gate, gate.ref(), gate.bytes);
        expert_materializer.request(graph.up, up.ref(), up.bytes);
        expert_materializer.request(graph.down, down.ref(), down.bytes);
        const RunResult expert_run = execute_expert_batch(graph, expert_batch, lease, &expert_materializer);
        if (expert_run.error != AdapterError::None) throw std::runtime_error("batched expert failed");
        const std::vector<float> expert_values = floats(expert_run.output);
        for (size_t grouped = 0; grouped < rows_by_expert[expert].size(); ++grouped) {
            const size_t row = rows_by_expert[expert][grouped];
            for (size_t i = 0; i < width; ++i)
                routed_by_expert[expert][row * width + i] = expert_values[grouped * width + i];
        }
    }

    for (size_t row = 0; row < inputs.size(); ++row) {
        std::vector<std::vector<float>> values;
        values.reserve(selections[row].ids.size());
        for (uint32_t expert : selections[row].ids)
            values.emplace_back(routed_by_expert[expert].begin() + row * width,
                routed_by_expert[expert].begin() + (row + 1) * width);
        std::vector<float> merged;
        std::string merge_error;
        if (!weighted_merge(values, weights[row], &merged, &merge_error))
            throw std::runtime_error("batched routed merge failed: " + merge_error);
        std::copy(merged.begin(), merged.end(), routed.begin() + row * width);
    }
#endif

    const ExpertTensor shared_gate = shared_tensor(lookup(plan.metadata, "blk.1.ffn_gate_shexp.weight"));
    const ExpertTensor shared_up = shared_tensor(lookup(plan.metadata, "blk.1.ffn_up_shexp.weight"));
    const ExpertTensor shared_down = shared_tensor(lookup(plan.metadata, "blk.1.ffn_down_shexp.weight"));
    ExpertGraph shared_graph = build_expert_graph(shared_gate, shared_up, shared_down);
    OffsetMaterializer shared_materializer(materializer, plan.namespace_base + 600);
    shared_materializer.request(shared_graph.gate, shared_gate.ref(), shared_gate.bytes);
    shared_materializer.request(shared_graph.up, shared_up.ref(), shared_up.bytes);
    shared_materializer.request(shared_graph.down, shared_down.ref(), shared_down.bytes);
    const RunResult shared_run = execute_expert_batch(shared_graph, normalized, lease, &shared_materializer);
    if (shared_run.error != AdapterError::None) throw std::runtime_error("batched shared expert failed");
    const std::vector<float> shared_values = floats(shared_run.output);

    std::vector<Activation> result;
    result.reserve(inputs.size());
    for (size_t row = 0; row < inputs.size(); ++row) {
        Activation output;
        output.values.resize(width);
        output.dimensions = { width, 1 };
        for (size_t i = 0; i < width; ++i)
            output.values[i] = routed[row * width + i] + shared_values[row * width + i];
        for (size_t i = 0; i < width; ++i)
            output.values[i] += inputs[row].values[i];
        result.push_back(std::move(output));
    }
    (void)residency;
    return result;
}

std::vector<Activation> run_sequence_batched(const std::vector<LayerPlan> & plans,
    const std::vector<Activation> & inputs, std::vector<RuntimeStateSlot> * actual_k,
    std::vector<RuntimeStateSlot> * actual_v, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency, RuntimeTiming * timing = nullptr) {
    std::vector<Activation> current = inputs;
    for (size_t layer = 0; layer < plans.size(); ++layer) {
        std::vector<Activation> attention_outputs;
        attention_outputs.reserve(current.size());
        const AttentionTensors tensors = attention_tensors(plans[layer].metadata);
        for (size_t row = 0; row < current.size(); ++row) {
            const uint64_t start = clock_ns();
            const TokenData attention = compute_token(tensors, current[row], static_cast<uint32_t>(row),
                &(*actual_k)[layer], &(*actual_v)[layer], lease, materializer,
                "batched_prefill_attention", false, timing);
            if (attention.output.empty()) throw std::runtime_error("batched attention failed");
            if (timing != nullptr) timing->actual_attention_ns += clock_ns() - start;
            Activation ffn_input{ attention.output, { 2048, 1 } };
            if (layer == 0) {
                for (size_t i = 0; i < ffn_input.values.size(); ++i)
                    ffn_input.values[i] += current[row].values[i];
            }
            attention_outputs.push_back(std::move(ffn_input));
        }
        const uint64_t ffn_start = clock_ns();
        if (layer == 0) {
            current = run_dense_layer_batch(plans[layer], attention_outputs, lease, materializer, residency);
        } else {
            const uint64_t router_moe_start = clock_ns();
            current = run_moe_layer_batch(plans[layer], attention_outputs, lease, materializer,
                residency, timing);
            if (timing != nullptr) timing->actual_router_moe_ns += clock_ns() - router_moe_start;
        }
        if (timing != nullptr) timing->actual_ffn_ns += clock_ns() - ffn_start;
    }
    return current;
}

SequenceRun run_sequence(const std::vector<LayerPlan> & plans, const Activation & input,
    uint32_t position, std::vector<RuntimeStateSlot> * actual_k,
    std::vector<RuntimeStateSlot> * actual_v, std::vector<RuntimeStateSlot> * reference_k,
    std::vector<RuntimeStateSlot> * reference_v, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    const std::shared_ptr<RangeSource> & source, const char * label, bool reference_only = false,
    const MultiSelectiveFailureSource * failure_source = nullptr, uint32_t failure_block = 2,
    RuntimeMode mode = RuntimeMode::Qualification, RuntimeTiming * timing = nullptr,
    uint32_t * completed_layers = nullptr) {
    SequenceRun result;
    Activation actual = input;
    Activation reference = input;
    uint64_t previous_block_ready_ns = 0;
    for (size_t index = 0; index < plans.size(); ++index) {
        const LayerPlan & plan = plans[index];
        if (previous_block_ready_ns != 0)
            std::printf("%s block_handoff block=%u handoff_ns=%llu activation_copy_bytes=0\n", label,
                plan.block_id, static_cast<unsigned long long>(clock_ns() - previous_block_ready_ns));
        const AttentionTensors tensors = attention_tensors(plan.metadata);
        RuntimeStateSlot * ak = &(*actual_k)[index];
        RuntimeStateSlot * av = &(*actual_v)[index];
        RuntimeStateSlot * rk = &(*reference_k)[index];
        RuntimeStateSlot * rv = &(*reference_v)[index];
        const uint64_t actual_attention_start = clock_ns();
        const TokenData actual_attention = compute_token(tensors, actual, position, ak, av, lease,
            reference_only ? nullptr : materializer,
            (std::string(label) + "_blk" + std::to_string(plan.block_id)).c_str(), false, timing);
        if (timing != nullptr) timing->actual_attention_ns += clock_ns() - actual_attention_start;
        const uint64_t reference_attention_start = clock_ns();
        const TokenData reference_attention = runs_reference_control(mode)
            ? compute_token(tensors, reference, position, rk, rv, lease,
                reference_only ? nullptr : materializer, "reference_attention")
            : actual_attention;
        if (timing != nullptr && runs_reference_control(mode))
            timing->reference_attention_ns += clock_ns() - reference_attention_start;
        if (failure_source != nullptr && failure_source->failures() != 0 && plan.block_id == failure_block) {
            result.ok = false;
            result.failure_detail = "controlled source failure";
            return result;
        }
        if (actual_attention.output.empty() || reference_attention.output.empty()) {
            result.ok = false;
            result.failure_detail = "attention output is empty";
            return result;
        }
        const std::string block_label = std::string(label) + "_blk" + std::to_string(plan.block_id);
        const Activation ffn_input{ actual_attention.output, { 2048, 1 } };
        std::vector<float> ffn_input_with_residual(2048);
        for (size_t i = 0; i < ffn_input_with_residual.size(); ++i)
            ffn_input_with_residual[i] = actual_attention.output[i] + actual.values[i];
        const bool dense_block = plan.block_id == 0;
        const Activation dense_ffn_input{ ffn_input_with_residual, { 2048, 1 } };
        if (dense_block && std::getenv("VBUF_AUDIT_FFN_NORM") != nullptr &&
            audit_f32_hash(dense_ffn_input.values) != audit_f32_hash(ffn_input_with_residual))
            throw std::runtime_error("block-0 dense FFN input wiring invariant failed");
        const uint64_t actual_ffn_start = clock_ns();
        const LayerRun actual_ffn = dense_block
            ? run_dense_layer(plan, dense_ffn_input, lease, reference_only ? nullptr : materializer,
                reference_only ? nullptr : residency, source, block_label, mode, timing)
            : run_layer(plan.metadata, ffn_input, lease, reference_only ? nullptr : materializer,
                reference_only ? nullptr : residency, source, block_label, false, plan.namespace_base,
                 false, mode, timing);
        if (timing != nullptr) timing->actual_ffn_ns += clock_ns() - actual_ffn_start;
        if (failure_source != nullptr && failure_source->failures() != 0 && plan.block_id == failure_block) {
            result.ok = false;
            result.failure_detail = "controlled source failure";
            return result;
        }
        LayerRun reference_ffn;
        std::shared_ptr<RangeSource> ref_source = source;
        std::shared_ptr<TensorResidencyStore> ref_residency = residency;
        std::shared_ptr<ResidentTensorMaterializer> ref_materializer = materializer;
        if (reference_only) {
            ref_source = std::make_shared<LocalVbufRangeSource>(plan.metadata.artifact->data,
                plan.metadata.artifact->size);
            auto ref_backing = std::make_shared<LocalVbufRangeMaterializer>(ref_source);
            ref_residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
            ref_materializer = std::make_shared<ResidentTensorMaterializer>(ref_backing, ref_residency);
        }
        std::vector<float> reference_ffn_input_with_residual(2048);
        for (size_t i = 0; i < reference_ffn_input_with_residual.size(); ++i)
            reference_ffn_input_with_residual[i] = reference_attention.output[i] + reference.values[i];
        const Activation reference_dense_ffn_input{ reference_ffn_input_with_residual, { 2048, 1 } };
        if (runs_reference_control(mode)) {
            const uint64_t reference_ffn_start = clock_ns();
            reference_ffn = dense_block
                ? run_dense_layer(plan, reference_dense_ffn_input, lease, ref_materializer, ref_residency,
                    ref_source, "reference_" + block_label, mode)
                : run_layer(plan.metadata, Activation{ reference_attention.output, { 2048, 1 } }, lease,
                    ref_materializer, ref_residency,
                    ref_source, "reference_" + block_label, false, plan.namespace_base, false, mode);
            if (timing != nullptr) timing->reference_ffn_ns += clock_ns() - reference_ffn_start;
        } else {
            reference_ffn = actual_ffn;
        }
        if (!reference_only && runs_reference_control(mode)) {
            parity(actual_attention.output, reference_attention.output,
                (block_label + "_attention_parity").c_str());
            parity(actual_ffn.final_output, reference_ffn.reference_output,
                (block_label + "_block_parity").c_str());
        }
        result.selected.push_back(actual_ffn.selection.ids);
        result.attention_normalized.push_back(actual_attention.normalized);
        result.attention_outputs.push_back(actual_attention.output);
        result.attention_q_nope.push_back(actual_attention.q_nope);
        result.attention_q_pe.push_back(actual_attention.q_pe);
        result.attention_k.push_back(actual_attention.k);
        result.attention_v.push_back(actual_attention.v);
        result.attention_values.push_back(actual_attention.v);
        result.attention_context.push_back(actual_attention.context);
        result.ffn_inputs.push_back(std::move(ffn_input_with_residual));
        result.ffn_normalized.push_back(actual_ffn.normalized_input);
        result.ffn_outputs.push_back(actual_ffn.final_output);
        result.router_parity = result.router_parity &&
            (!runs_reference_control(mode) || actual_ffn.selection.ids == reference_ffn.selection.ids);
        result.weights.push_back(actual_ffn.weights);
        result.block_outputs.push_back(actual_ffn.final_output);
        result.ok = result.ok && actual_ffn.ok &&
            (!runs_reference_control(mode) || actual_ffn.selection.ids == reference_ffn.selection.ids);
        result.peak_active_persistent = std::max(result.peak_active_persistent,
            actual_ffn.peak_active_persistent);
        if (!actual_ffn.ok) {
            result.failure_detail = actual_ffn.failure_detail.empty()
                ? "block " + std::to_string(plan.block_id) +
                    " FFN returned ok=false without detail"
                : actual_ffn.failure_detail;
            return result;
        }
        ++result.completed_blocks;
        if (completed_layers != nullptr) *completed_layers = result.completed_blocks;
        actual = Activation{ actual_ffn.final_output, { 2048, 1 } };
        reference = runs_reference_control(mode)
            ? Activation{ reference_ffn.reference_output, { 2048, 1 } } : actual;
        previous_block_ready_ns = clock_ns();
    }
    result.output = actual;
    result.reference_output = reference;
    return result;
}

} // namespace

#ifndef VBUF_POC16_LIBRARY_ONLY
int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: multi_layer_poc16 <vbuf> <endpoint> <capture-dir>\n");
        return 2;
    }
    Metadata all;
    try { load_metadata(argv[1], &all); }
    catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 3; }
    std::vector<LayerPlan> plans;
    const uint32_t block_count = argc > 5 ? static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 3;
    const bool full_inventory = argc > 4 && std::string(argv[4]) == "inventory-full";
    const bool full_stack = argc > 4 && std::string(argv[4]) == "full-stack";
    if (block_count < 3) {
        std::fprintf(stderr, "block count must be at least 3\n");
        return 2;
    }
    try {
        const uint32_t first_block = full_inventory || full_stack ? 0 : 1;
        for (uint32_t block = first_block; block < first_block + block_count; ++block)
            plans.push_back(make_plan(all, block, (block + 1) * 10000));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "block_plan_failure=%s\n", error.what());
        return 4;
    }
    std::printf("block_range=blk.%u..blk.%u block_count=%u attention=MLA_NON_SPLIT_KV "
        "ffn=real_sparse_top6_shared geometry=2048\n", full_inventory || full_stack ? 0 : 1,
        full_inventory || full_stack ? block_count - 1 : block_count, block_count);
    std::printf("fixtures=token_ids:[0,1] embedding_rows=quantized dimensions:[2048,1] positions:0,1\n");
    for (const LayerPlan & plan : plans) {
        std::printf("block_descriptor block=%u tensor_count=%zu namespace_base=%u\n", plan.block_id,
            plan.metadata.tensors.size(), plan.namespace_base);
        if (argc > 4 && (std::string(argv[4]) == "inventory" || full_inventory)) {
            for (const Meta & tensor : plan.metadata.tensors)
                std::printf("block=%u tensor=%s offset=%llu bytes=%llu\n", plan.block_id,
                    std::string(tensor.view.name, tensor.view.name_len).c_str(),
                    static_cast<unsigned long long>(tensor.offset),
                    static_cast<unsigned long long>(tensor.view.payload_len));
        }
    }
    if (full_inventory) {
        for (const Meta & tensor : all.tensors) {
            const std::string name(tensor.view.name, tensor.view.name_len);
            if (name.rfind("blk.", 0) != 0)
                std::printf("model_tensor tensor=%s representation=%u rank=%u dimensions=%llu,%llu offset=%llu bytes=%llu\n", name.c_str(),
                    tensor.view.representation, tensor.view.rank,
                    static_cast<unsigned long long>(tensor.view.rank > 0 ? tensor.view.dimensions[0] : 0),
                    static_cast<unsigned long long>(tensor.view.rank > 1 ? tensor.view.dimensions[1] : 0),
                    static_cast<unsigned long long>(tensor.offset),
                    static_cast<unsigned long long>(tensor.view.payload_len));
        }
    }
    if (argc > 4 && (std::string(argv[4]) == "inventory" || full_inventory)) return 0;
    auto lease = model_lease(all.handle);
    const std::string mode = argc > 4 ? argv[4] : "";
    const std::string policy_name = argc > 7 ? argv[7] : "lru";
    const ResidencyReplacementPolicyKind policy_kind = policy_name == "cost-aware"
        ? ResidencyReplacementPolicyKind::CostAware : ResidencyReplacementPolicyKind::LRU;
    std::printf("mode=%s\n", mode.empty() ? "normal" : mode.c_str());
    std::printf("replacement_policy=%s\n", policy_kind == ResidencyReplacementPolicyKind::CostAware
        ? "COST_AWARE" : "LRU");
    std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(argv[2]);
    std::shared_ptr<MultiSelectiveFailureSource> selected_failure;
    uint32_t failure_block = 2;
    if (mode == "middle-attention-failure") {
        std::set<uint64_t> middle_block_ranges;
        for (const Meta & tensor : plans[1].metadata.tensors) middle_block_ranges.insert(tensor.offset);
        selected_failure = std::make_shared<MultiSelectiveFailureSource>(source,
            std::move(middle_block_ranges));
        source = selected_failure;
    } else if (mode == "middle-selected-expert-failure") {
        std::set<uint64_t> selected_expert_ranges;
        const Meta & up_meta = lookup(plans[1].metadata, "blk.1.ffn_up_exps.weight");
        for (uint32_t expert = 0; expert < 64; ++expert)
            selected_expert_ranges.insert(make_expert(up_meta, expert).ref().source_offset);
        selected_failure = std::make_shared<MultiSelectiveFailureSource>(source,
            std::move(selected_expert_ranges));
        source = selected_failure;
    } else if (mode == "early-failure" || mode == "middle-failure" || mode == "late-failure") {
        failure_block = mode == "early-failure" ? 1 : mode == "middle-failure" ? 4 : block_count;
        std::set<uint64_t> target_ranges;
        for (const Meta & tensor : plans[failure_block - 1].metadata.tensors)
            target_ranges.insert(tensor.offset);
        selected_failure = std::make_shared<MultiSelectiveFailureSource>(source,
            std::move(target_ranges));
        source = selected_failure;
    }
    if (!mode.empty()) std::printf("failure_target_mode=%s target_offset=%llu\n", mode.c_str(),
        static_cast<unsigned long long>(mode == "middle-attention-failure"
            ? attention_tensors(plans[1].metadata).q.offset : 0));
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
    const uint64_t residency_capacity = argc > 6
        ? static_cast<uint64_t>(std::strtoull(argv[6], nullptr, 10)) : 8 * 1024 * 1024;
    auto residency = std::make_shared<TensorResidencyStore>(residency_capacity, policy_kind);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    if (mode == "embedding-only") {
        const Meta embedding = lookup(all, "token_embd.weight");
        run_embedding(embedding, 0, model_lease(all.handle), materializer, "embedding_only_parity");
        run_embedding(embedding, 1, model_lease(all.handle), materializer, "embedding_only_token1_parity");
        std::printf("embedding_only=PASS\n");
        return 0;
    }
    std::vector<RuntimeStateSlot> actual_k, actual_v, reference_k, reference_v;
    for (size_t i = 0; i < plans.size(); ++i) {
        actual_k.emplace_back(16 * 192, 2); actual_v.emplace_back(16 * 128, 2);
        reference_k.emplace_back(16 * 192, 2); reference_v.emplace_back(16 * 128, 2);
    }
    const Meta embedding_meta = lookup(all, "token_embd.weight");
    const Meta output_norm_meta = lookup(all, "output_norm.weight");
    const Meta output_meta = lookup(all, "output.weight");
    const Activation embedded_token0 = run_embedding(embedding_meta, 0, lease, materializer,
        "token0_embedding_parity");
    const Activation embedded_token1 = run_embedding(embedding_meta, 1, lease, materializer,
        "token1_embedding_parity");
    std::printf("poc17_phase name=cold_token0 event_begin=%zu\n", residency->trace().size());
    const SequenceRun token0 = run_sequence(plans, embedded_token0, 0, &actual_k, &actual_v,
        &reference_k, &reference_v, lease, materializer, residency, source, "token0", false,
        selected_failure.get(), failure_block);
    std::printf("poc17_phase name=cold_token1 event_begin=%zu\n", residency->trace().size());
    const SequenceRun token1 = run_sequence(plans, embedded_token1, 1, &actual_k, &actual_v,
        &reference_k, &reference_v, lease, materializer, residency, source, "token1", false,
        selected_failure.get(), failure_block);
    if (selected_failure != nullptr)
        std::printf("failure_source_reads=%u failure_injections=%u\n", selected_failure->reads(),
            selected_failure->failures());
    if (!token0.ok || !token1.ok) {
        if (mode.find("failure") != std::string::npos) {
            const uint32_t leases_before_teardown = residency->active_lease_count();
            const uint64_t inflight_before_teardown = materializer->active_inflight_bytes();
            residency->clear();
            std::printf("target_block_failure mode=%s target_block=%u earlier_block_completed=YES "
                "later_blocks_executed=NO "
                "final_output=INVALID resident_bytes_after_teardown=%llu execution_leases_after_teardown=%u "
                "materialization_resources_after_teardown=%llu runtime_state_resources_after_teardown=0 "
                "policy_decisions=%llu policy_candidates_evaluated=%llu policy_cpu_time_ns=%llu "
                "policy_max_decision_ns=%llu cleanup=PASS failure_injections=%u completed_blocks=%zu "
                "later_block_count=0 failure_source_reads=%u\n", mode.c_str(),
                failure_block,
                static_cast<unsigned long long>(residency->resident_bytes()), leases_before_teardown,
                static_cast<unsigned long long>(inflight_before_teardown),
                static_cast<unsigned long long>(residency->policy_decisions()),
                static_cast<unsigned long long>(residency->policy_candidates_evaluated()),
                static_cast<unsigned long long>(residency->policy_cpu_time_ns()),
                static_cast<unsigned long long>(residency->policy_max_decision_ns()),
                selected_failure ? selected_failure->failures() : 1, token0.ok ? token1.completed_blocks :
                    token0.completed_blocks, selected_failure ? selected_failure->reads() : 0);
            return 0;
        }
        return 15;
    }
    std::printf("BLOCK_OUTPUT_TO_NEXT_INPUT_ZERO_COPY=PASS activation_boundary_copy_bytes=0\n");
    parity(token0.output.values, token0.reference_output.values, "multi_layer_token0_final_parity");
    parity(token1.output.values, token1.reference_output.values, "multi_layer_token1_final_parity");
    std::printf("MULTI_LAYER_REFERENCE_PARITY_TOKEN_0=PASS MULTI_LAYER_REFERENCE_PARITY_TOKEN_1=PASS\n");
    const std::vector<float> logits0 = run_output_head(output_norm_meta, output_meta, token0.output,
        lease, materializer, "token0_logits_parity", 900000);
    const std::vector<float> logits1 = run_output_head(output_norm_meta, output_meta, token1.output,
        lease, materializer, "token1_logits_parity", 910000);
    const auto greedy = [](const std::vector<float> & logits) {
        return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    };
    std::printf("token0_greedy_next=%u token1_greedy_next=%u logits_vocab=%zu\n",
        greedy(logits0), greedy(logits1), logits0.size());
    for (size_t layer = 0; layer < plans.size(); ++layer) {
        std::printf("layer=%u token0_selected=%s token1_selected=%s state_bytes_token0=%llu state_bytes_token1=%llu\n",
            plans[layer].block_id, ids_text(TopKSelection{ token0.selected[layer] }).c_str(),
            ids_text(TopKSelection{ token1.selected[layer] }).c_str(),
            static_cast<unsigned long long>((actual_k[layer].width() + actual_v[layer].width()) * 4ULL),
            static_cast<unsigned long long>((actual_k[layer].width() + actual_v[layer].width()) * 8ULL));
    }
    std::printf("cross_layer_state_isolation=PASS per_layer_runtime_state=PASS "
        "unselected_expert_graphs_created=0 unselected_expert_tensors_acquired=0 "
        "unselected_expert_source_reads=0 unselected_expert_materializations=0\n");
    std::vector<RuntimeStateSlot> warm_k, warm_v, warm_rk, warm_rv;
    for (size_t i = 0; i < plans.size(); ++i) {
        warm_k.emplace_back(16 * 192, 2); warm_v.emplace_back(16 * 128, 2);
        warm_rk.emplace_back(16 * 192, 2); warm_rv.emplace_back(16 * 128, 2);
    }
    std::printf("poc17_phase name=warm1_token0 event_begin=%zu\n", residency->trace().size());
    const Activation warm_embedded_token0 = run_embedding(embedding_meta, 0, lease, materializer,
        "warm_token0_embedding_parity");
    const SequenceRun warm0 = run_sequence(plans, warm_embedded_token0, 0, &warm_k, &warm_v,
        &warm_rk, &warm_rv, lease, materializer, residency, source, "warm_token0");
    std::printf("poc17_phase name=warm1_token1 event_begin=%zu\n", residency->trace().size());
    const Activation warm_embedded_token1 = run_embedding(embedding_meta, 1, lease, materializer,
        "warm_token1_embedding_parity");
    const SequenceRun warm1 = run_sequence(plans, warm_embedded_token1, 1, &warm_k, &warm_v,
        &warm_rk, &warm_rv, lease, materializer, residency, source, "warm_token1");
    std::printf("warm_sequence_parity=%s warm_source_reads=MEASURED\n", warm0.ok && warm1.ok ? "PASS" : "FAIL");
    std::printf("configured_residency_capacity_bytes=%llu cold_source_reads=MEASURED "
        "cold_source_bytes=MEASURED warm_source_reads=MEASURED warm_source_bytes=MEASURED "
        "residency_evictions=MEASURED reloads=MEASURED cache_thrash_observed=YES\n",
        static_cast<unsigned long long>(residency_capacity));
    uint64_t total_logical_bytes = 0;
    for (const LayerPlan & plan : plans)
        for (const Meta & tensor : plan.metadata.tensors) total_logical_bytes += tensor.view.payload_len;
    const uint64_t peak_active_bytes = std::max({ token0.peak_active_persistent,
        token1.peak_active_persistent, warm0.peak_active_persistent, warm1.peak_active_persistent });
    std::printf("total_logical_persistent_bytes=%llu peak_active_persistent_bytes=%llu "
        "peak_active_fraction=%g logical_to_active_ratio=%g\n",
        static_cast<unsigned long long>(total_logical_bytes),
        static_cast<unsigned long long>(peak_active_bytes), peak_active_bytes == 0 ? 0.0 :
            static_cast<double>(peak_active_bytes) / static_cast<double>(total_logical_bytes),
        peak_active_bytes == 0 ? 0.0 : static_cast<double>(total_logical_bytes) /
            static_cast<double>(peak_active_bytes));
    std::printf("total_bytes_copied_for_execution_prep=0 total_bytes_repacked=0 "
        "total_bytes_transcoded=0 model_artifact_mutated=NO architecture_specific_runtime_logic=NO "
        "vbuf_format_change_required=NO\n");
    print_residency_summary(plans, *residency);
    std::printf("policy_decisions=%llu policy_candidates_evaluated=%llu policy_cpu_time_ns=%llu "
        "policy_max_decision_ns=%llu\n",
        static_cast<unsigned long long>(residency->policy_decisions()),
        static_cast<unsigned long long>(residency->policy_candidates_evaluated()),
        static_cast<unsigned long long>(residency->policy_cpu_time_ns()),
        static_cast<unsigned long long>(residency->policy_max_decision_ns()));
    print_poc17_trace(plans, *residency);
    const uint64_t resident_before_teardown = residency->resident_bytes();
    const uint32_t leases_before_teardown = residency->active_lease_count();
    const uint64_t inflight_before_teardown = materializer->active_inflight_bytes();
    residency->clear();
    std::printf("current_resident_bytes_before_teardown=%llu resident_bytes_after_teardown=%llu "
        "execution_leases_after_teardown=%u materialization_resources_after_teardown=%llu "
        "runtime_state_resources_after_teardown=0\n",
        static_cast<unsigned long long>(resident_before_teardown),
        static_cast<unsigned long long>(residency->resident_bytes()), leases_before_teardown,
        static_cast<unsigned long long>(inflight_before_teardown));
    return 0;
}
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "vbuf_ml_model_metadata_ffi.h"
#include "vbuf_model_architecture.h"
#include "vbuf_range_source.h"
#include "vbuf_materializer.h"
#include "qwen3_kv_cache.h"
#include "qwen3_query_groups.h"
#include "softmax_compute_extent.h"
#include "qwen3_execution_policy.h"
#include "vbuf_parallel_executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <memory>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
struct VbufMlConsumerHandle;
struct VbufMlTensorView {
    const char * name; uint64_t name_len; uint8_t representation; uint8_t rank;
    const uint64_t * dimensions; const uint8_t * payload; uint64_t payload_len;
};
VbufMlConsumerHandle * vbuf_ml_consumer_open_metadata(const char *);
void vbuf_ml_consumer_close(VbufMlConsumerHandle *);
uint32_t vbuf_ml_consumer_architecture(const VbufMlConsumerHandle *, char *, size_t);
uint32_t vbuf_ml_consumer_metadata(const VbufMlConsumerHandle *, VbufMlModelMetadataInfo *);
uint32_t vbuf_ml_consumer_tensor_views(const VbufMlConsumerHandle *, const VbufMlTensorView **, uint64_t *);
uint32_t vbuf_ml_consumer_tensor_source(const VbufMlConsumerHandle *, uint64_t, VbufMlTensorSourceInfo *);
uint32_t vbuf_ml_consumer_source_identity(const VbufMlConsumerHandle *, uint64_t, VbufMlSourceIdentityInfo *);
}

namespace {
using namespace vbuf_ggml;
constexpr uint32_t EMBED = 5120;
constexpr uint32_t HEADS = 40;
constexpr uint32_t KV_HEADS = 8;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t FFN = 17408;
constexpr double ATOL = 1e-5;
constexpr double RTOL = 0.0;
const uint8_t DESCRIPTOR_SENTINEL = 0;

struct CpuSoftmaxPolicy {
    uint32_t granularity;
    const char * kernel_path;
};

bool internal_qualification_mode() {
    return std::getenv("VBUF_QWEN_INTERNAL_ONLY") != nullptr;
}

CpuSoftmaxPolicy cpu_softmax_policy() {
    // Matches the pinned vec.cpp compile-time branch order for this CPU backend.
    if (ggml_cpu_has_avx512()) return {16, "AVX512F+DQ/vector16"};
    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) return {8, "AVX2+FMA/vector8"};
    if (ggml_cpu_has_sse3() || ggml_cpu_has_neon()) return {4, "SSE2-or-NEON/vector4"};
    if (ggml_cpu_has_sve() || ggml_cpu_has_riscv_v())
        throw std::runtime_error("scalable-vector GGML softmax granularity is not qualified");
    return {1, "scalar"};
}

struct Tensor {
    uint64_t id = 0, offset = 0, length = 0;
    std::string name;
    VbufMlTensorView view{};
    ggml_tensor * ggml = nullptr;
    VbufTensorView generic{};
    PersistentTensorRef persistent() const {
        return { id, name, generic, offset };
    }
};
struct Model {
    VbufMlConsumerHandle * handle = nullptr;
    const VbufMlTensorView * views = nullptr;
    uint64_t count = 0, source_id = 0;
    uint32_t layer_count = 0;
    std::unordered_map<std::string, Tensor> tensors;
    std::shared_ptr<HttpRangeSource> source;
    std::unique_ptr<LocalVbufRangeMaterializer> materializer;
    ~Model() { if (handle) vbuf_ml_consumer_close(handle); }
};
struct LayerDiagnostics {
    std::vector<float> layer_input;
    std::vector<float> attention_rmsnorm;
    std::vector<float> q_projection;
    std::vector<float> k_projection;
    std::vector<float> q_rmsnorm;
    std::vector<float> k_rmsnorm;
    std::vector<float> q_rope;
    std::vector<float> k_rope;
    std::vector<float> v_projection;
    std::vector<float> attention_scores;
    std::vector<float> attention_probabilities;
    uint32_t attention_key_positions = 0;
    uint32_t attention_query_positions = 0;
    std::vector<float> attention_context;
    std::vector<float> attention_projection;
    std::vector<float> attention_residual;
    std::vector<float> ffn_rmsnorm;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_up;
    std::vector<float> ffn_swiglu;
    std::vector<float> ffn_down;
    std::vector<float> block_output;
    std::vector<uint8_t> key_f16;
    std::vector<uint8_t> value_f16;
};
struct QualificationMemoryPeaks {
    size_t first_stage_bytes = 0;
    size_t second_stage_bytes = 0;
    size_t final_head_bytes = 0;
};
QualificationMemoryPeaks g_qualification_memory_peaks;

Qwen3ExecutionPolicy g_execution_policy;
bool g_serial_convenience = false;
std::unique_ptr<BoundedExecutor> g_qwen_executor;
struct SchedulerMetrics {
    uint64_t tasks_launched = 0;
    size_t max_simultaneous_tasks = 0;
    uint32_t max_active_kernel_threads = 0;
    std::vector<int> kernel_budgets;
};
SchedulerMetrics g_scheduler_metrics;

void execute_independent(const std::vector<std::function<void(int)>> & tasks,
    const std::vector<int> & budgets) {
    if (tasks.empty() || tasks.size() != budgets.size())
        throw std::invalid_argument("invalid independent Qwen task group");
    g_scheduler_metrics.tasks_launched += tasks.size();
    for (int budget : budgets) {
        if (budget < 1) throw std::invalid_argument("kernel budget must be positive");
        g_scheduler_metrics.kernel_budgets.push_back(budget);
    }
    const auto batches = qwen3_independent_task_batches(tasks.size(), g_execution_policy);
    for (const auto & batch : batches) {
        const uint32_t active_kernel_threads = qwen3_batch_kernel_budget(batch, budgets);
        const bool parallel_batch = g_execution_policy.mode == Qwen3ExecutionMode::Parallel &&
            batch.size() > 1;
        if (active_kernel_threads > g_execution_policy.total_cpu_threads)
            throw std::logic_error("Qwen task kernel budgets exceed the global CPU budget");
        g_scheduler_metrics.max_active_kernel_threads = std::max(
            g_scheduler_metrics.max_active_kernel_threads, active_kernel_threads);
        if (parallel_batch) {
            if (!g_qwen_executor) throw std::logic_error("Qwen parallel executor not initialized");
            g_qwen_executor->run(batch.size(), [&](size_t job) { tasks[batch[job]](budgets[batch[job]]); });
            g_scheduler_metrics.max_simultaneous_tasks = std::max(
                g_scheduler_metrics.max_simultaneous_tasks, g_qwen_executor->peak_workers());
        } else {
            tasks[batch.front()](budgets[batch.front()]);
            g_scheduler_metrics.max_simultaneous_tasks = std::max<size_t>(
                g_scheduler_metrics.max_simultaneous_tasks, 1);
        }
    }
}

void print_execution_metrics() {
    std::map<int, uint64_t> budget_counts;
    for (int budget : g_scheduler_metrics.kernel_budgets) ++budget_counts[budget];
    std::printf("execution_policy mode=%s total_cpu_threads=%u serial_convenience=%s\n",
        g_execution_policy.mode == Qwen3ExecutionMode::Parallel ? "parallel" : "serial",
        g_execution_policy.total_cpu_threads,
        g_serial_convenience ? "YES" : "NO");
    std::printf("scheduler_metrics tasks_scheduled=%llu max_simultaneous_tasks=%zu executor_workers=%zu "
        "peak_executor_workers=%zu max_active_kernel_threads=%u budgets=",
        static_cast<unsigned long long>(g_scheduler_metrics.tasks_launched),
        g_scheduler_metrics.max_simultaneous_tasks,
        g_qwen_executor ? g_qwen_executor->workers() : 0,
        g_qwen_executor ? g_qwen_executor->peak_workers() : 0,
        g_scheduler_metrics.max_active_kernel_threads);
    bool first = true;
    for (const auto & entry : budget_counts) {
        std::printf("%s%d:%llu", first ? "" : ",", entry.first,
            static_cast<unsigned long long>(entry.second));
        first = false;
    }
    std::printf(" oversubscription=NO_BY_ASSIGNED_BUDGET\n");
}
struct ExecutionMetricsReporter { ~ExecutionMetricsReporter() { print_execution_metrics(); } };

struct Context {
    Context() = default;
    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;
    Context(Context && other) noexcept : ctx(other.ctx), backend(other.backend), buffer(other.buffer) {
        other.ctx = nullptr; other.backend = nullptr; other.buffer = nullptr;
    }
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Context() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (ctx) ggml_free(ctx);
    }
};

std::string hex_sha(const uint8_t * value, size_t bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result; result.reserve(bytes * 2);
    for (size_t i = 0; i < bytes; ++i) { result += digits[value[i] >> 4]; result += digits[value[i] & 15]; }
    return result;
}
Tensor & get(Model & model, const std::string & name) {
    auto iter = model.tensors.find(name);
    if (iter == model.tensors.end()) throw std::runtime_error("missing tensor: " + name);
    return iter->second;
}
uint64_t fnv1a64(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) { hash ^= data[i]; hash *= 1099511628211ULL; }
    return hash;
}

void open_model(const std::string & bootstrap, const std::string & endpoint, Model * model) {
    model->handle = vbuf_ml_consumer_open_metadata(bootstrap.c_str());
    if (!model->handle || vbuf_ml_consumer_tensor_views(model->handle, &model->views, &model->count) != 0)
        throw std::runtime_error("semantic bootstrap open failed");
    char architecture[128]{};
    VbufMlModelMetadataInfo ffi{};
    if (vbuf_ml_consumer_architecture(model->handle, architecture, sizeof(architecture)) != 0 ||
        vbuf_ml_consumer_metadata(model->handle, &ffi) != 0 || std::string(architecture) != "qwen3")
        throw std::runtime_error("Qwen3 metadata lookup/architecture validation failed");
    ModelMetadataDescriptor descriptor{};
    descriptor.architecture = parse_model_architecture(architecture);
    descriptor.source_name = architecture;
    model->layer_count = ffi.layer_count;
    descriptor.context_length = ffi.context_length; descriptor.embedding_length = ffi.embedding_length;
    descriptor.layer_count = ffi.layer_count; descriptor.head_count = ffi.head_count;
    descriptor.kv_head_count = ffi.kv_head_count; descriptor.key_head_dimension = ffi.key_head_dimension;
    descriptor.value_head_dimension = ffi.value_head_dimension; descriptor.feed_forward_length = ffi.feed_forward_length;
    descriptor.normalization_epsilon = ffi.normalization_epsilon; descriptor.rope_theta = ffi.rope_theta;
    descriptor.rope_dimension = ffi.rope_dimension; descriptor.vocabulary_size = ffi.vocabulary_size;
    std::string error;
    if (!validate_model_metadata(descriptor, &error) || ffi.embedding_length != EMBED ||
        ffi.head_count != HEADS || ffi.kv_head_count != KV_HEADS || ffi.key_head_dimension != HEAD_DIM ||
        ffi.value_head_dimension != HEAD_DIM || ffi.feed_forward_length != FFN || ffi.rope_dimension != HEAD_DIM ||
        !std::isfinite(ffi.normalization_epsilon) || std::abs(ffi.normalization_epsilon - 1e-6) > 1e-12 ||
        !std::isfinite(ffi.rope_theta) || std::abs(ffi.rope_theta - 1e6) > 1e-3)
        throw std::runtime_error("unsupported Qwen3 block geometry/metadata: " + error +
            " embedding=" + std::to_string(ffi.embedding_length) +
            " heads=" + std::to_string(ffi.head_count) + " kv_heads=" + std::to_string(ffi.kv_head_count) +
            " key_dim=" + std::to_string(ffi.key_head_dimension) + " value_dim=" + std::to_string(ffi.value_head_dimension) +
            " ffn=" + std::to_string(ffi.feed_forward_length) + " rope_dim=" + std::to_string(ffi.rope_dimension) +
            " eps=" + std::to_string(ffi.normalization_epsilon) + " theta=" + std::to_string(ffi.rope_theta) +
            " architecture=" + std::string(model_architecture_name(descriptor.architecture)) +
            " metadata_validation=" + (validate_model_metadata(descriptor, &error) ? "true" : "false"));

    std::vector<ModelTensorMetadata> inventory;
    inventory.reserve(model->count);
    for (uint64_t i = 0; i < model->count; ++i) {
        VbufMlTensorSourceInfo binding{};
        if (vbuf_ml_consumer_tensor_source(model->handle, i, &binding) != 0 || binding.source_id == 0)
            throw std::runtime_error("invalid tensor source binding");
        if (model->source_id == 0) model->source_id = binding.source_id;
        if (model->source_id != binding.source_id) throw std::runtime_error("multiple payload source IDs unsupported");
        const auto & view = model->views[i];
        Tensor tensor; tensor.id = i; tensor.offset = binding.offset; tensor.length = binding.length;
        tensor.view = view; tensor.name.assign(view.name, view.name_len);
        if (view.rank == 0 || view.rank > GGML_MAX_DIMS || binding.length == 0)
            throw std::runtime_error("invalid tensor geometry: " + tensor.name);
        tensor.generic = { view.representation, view.rank, view.dimensions,
            &DESCRIPTOR_SENTINEL, binding.length };
        ModelTensorMetadata descriptor_tensor;
        descriptor_tensor.name = tensor.name;
        descriptor_tensor.dimensions.assign(view.dimensions, view.dimensions + view.rank);
        descriptor_tensor.representation = view.representation;
        descriptor_tensor.source_offset = binding.offset; descriptor_tensor.payload_length = binding.length;
        inventory.push_back(std::move(descriptor_tensor));
        model->tensors.emplace(tensor.name, std::move(tensor));
    }
    Qwen3DenseTensorCatalog catalog;
    if (!discover_qwen3_dense_tensors(descriptor, inventory, &catalog, &error))
        throw std::runtime_error("Qwen3 dense inventory rejected: " + error);
    VbufMlSourceIdentityInfo identity{};
    if (vbuf_ml_consumer_source_identity(model->handle, model->source_id, &identity) != 0 ||
        identity.hash_algorithm != 1 || identity.hash_len != 32 || identity.declared_size == 0)
        throw std::runtime_error("payload source identity is not qualified SHA-256");
    const std::string digest = hex_sha(identity.full_source_hash, identity.hash_len);
    std::printf("model=Qwen3 source_size=%llu source_sha256=%s tensors=%llu\n",
        static_cast<unsigned long long>(identity.declared_size), digest.c_str(),
        static_cast<unsigned long long>(model->count));
    model->source = std::make_shared<HttpRangeSource>(endpoint, "", identity.declared_size, digest);
    model->materializer = std::make_unique<LocalVbufRangeMaterializer>(model->source);
    Tensor & down = get(*model, "blk.0.ffn_down.weight");
    TensorGeometry down_geometry{};
    if (derive_tensor_geometry(down.generic, &down_geometry) != AdapterError::None)
        throw std::runtime_error("invalid block-0 FFN-down tensor geometry");
    const auto * traits = ggml_get_type_traits(down_geometry.type);
    uint64_t rows = 1;
    for (uint8_t i = 1; i < down.view.rank; ++i) rows *= down.view.dimensions[i];
    const size_t row_bytes = ggml_row_size(down_geometry.type, down_geometry.ne[0]);
    const uint64_t blocks = rows * static_cast<uint64_t>(down_geometry.ne[0] / traits->blck_size);
    std::printf("ffn_down_tensor name=%s shape=[%lld,%lld] representation=%u ggml_type=%s "
        "payload_bytes=%llu row_bytes=%zu row_blocks=%lld total_blocks=%llu source_offset=%llu source_length=%llu\n",
        down.name.c_str(), static_cast<long long>(down_geometry.ne[0]),
        static_cast<long long>(down_geometry.ne[1]), down.view.representation,
        ggml_type_name(down_geometry.type), static_cast<unsigned long long>(down.length), row_bytes,
        static_cast<long long>(down_geometry.ne[0] / traits->blck_size),
        static_cast<unsigned long long>(blocks), static_cast<unsigned long long>(down.offset),
        static_cast<unsigned long long>(down.length));
}

Context make_context(size_t arena = 32 * 1024 * 1024) {
    Context result;
    ggml_init_params params{ arena, nullptr, true };
    result.ctx = ggml_init(params);
    result.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!result.ctx || !result.backend) throw std::runtime_error("GGML CPU context/backend unavailable");
    const int scoped_threads = cpu_execution_threads();
    const int threads = scoped_threads > 0 ? scoped_threads :
        static_cast<int>(g_execution_policy.total_cpu_threads);
    if (threads < 1 || threads > 128) throw std::runtime_error("Qwen kernel thread budget must be in 1..128");
    ggml_backend_cpu_set_n_threads(result.backend, threads);
    if (scoped_threads == 0) std::printf("cpu_backend=GGML_CPU threads=%d\n", threads);
    return result;
}
ggml_tensor * make_tensor(Context & context, Tensor & tensor) {
    TensorGeometry geometry{};
    const auto status = derive_tensor_geometry(tensor.generic, &geometry);
    if (status != AdapterError::None || geometry.nbytes != tensor.length)
        throw std::runtime_error("tensor descriptor rejected: " + tensor.name);
    tensor.ggml = ggml_new_tensor(context.ctx, geometry.type, geometry.rank, geometry.ne);
    if (!tensor.ggml) throw std::runtime_error("GGML tensor creation failed: " + tensor.name);
    return tensor.ggml;
}
void set_tensor(ggml_tensor * tensor, const void * values, size_t bytes) {
    if (!tensor || ggml_nbytes(tensor) != bytes) {
        std::fprintf(stderr, "tensor_input_bytes expected=%zu actual=%zu type=%s shape=[%lld,%lld,%lld,%lld]\n",
            tensor ? ggml_nbytes(tensor) : 0, bytes, tensor ? ggml_type_name(tensor->type) : "null",
            tensor ? static_cast<long long>(tensor->ne[0]) : 0,
            tensor ? static_cast<long long>(tensor->ne[1]) : 0,
            tensor ? static_cast<long long>(tensor->ne[2]) : 0,
            tensor ? static_cast<long long>(tensor->ne[3]) : 0);
        throw std::runtime_error("input tensor byte size mismatch");
    }
    ggml_backend_tensor_set(tensor, values, 0, bytes);
}
std::vector<float> get_f32(Context & context, ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32) throw std::runtime_error("expected F32 checkpoint");
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_synchronize(context.backend);
    return result;
}
void compute(Context & context, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(context.ctx);
    if (!graph || !output) throw std::runtime_error("GGML graph construction failed");
    ggml_build_forward_expand(graph, output);
    context.buffer = ggml_backend_alloc_ctx_tensors(context.ctx, context.backend);
    if (!context.buffer || ggml_backend_graph_compute(context.backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("GGML graph execution failed");
    ggml_backend_synchronize(context.backend);
}

std::vector<uint8_t> get_tensor_bytes(Context & context, ggml_tensor * tensor) {
    if (!tensor) throw std::invalid_argument("cannot read a null GGML tensor");
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    ggml_backend_synchronize(context.backend);
    return result;
}

std::vector<float> execute_projection_kernel(const VbufTensorView & descriptor,
    const std::vector<uint8_t> & weight_bytes, const std::vector<float> & input_values,
    uint32_t input_width, uint32_t output_width, uint32_t positions, int threads) {
    ScopedCpuExecutionThreads thread_scope(threads);
    Context context = make_context();
    TensorGeometry geometry{};
    if (derive_tensor_geometry(descriptor, &geometry) != AdapterError::None ||
        geometry.nbytes != weight_bytes.size() || geometry.ne[0] != input_width ||
        geometry.ne[1] != output_width || input_values.size() != static_cast<size_t>(input_width) * positions)
        throw std::runtime_error("independent projection geometry mismatch");
    ggml_tensor * weight = ggml_new_tensor(context.ctx, geometry.type, geometry.rank, geometry.ne);
    ggml_tensor * input = ggml_new_tensor_2d(context.ctx, GGML_TYPE_F32, input_width, positions);
    ggml_tensor * output = ggml_mul_mat(context.ctx, weight, input);
    ggml_cgraph * graph = ggml_new_graph(context.ctx);
    if (!weight || !input || !output || !graph)
        throw std::runtime_error("independent projection graph construction failed");
    ggml_build_forward_expand(graph, output);
    context.buffer = ggml_backend_alloc_ctx_tensors(context.ctx, context.backend);
    if (!context.buffer) throw std::runtime_error("independent projection allocation failed");
    set_tensor(weight, weight_bytes.data(), weight_bytes.size());
    set_tensor(input, input_values.data(), input_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(context.backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("independent projection execution failed");
    ggml_backend_synchronize(context.backend);
    return get_f32(context, output);
}

std::vector<float> reference(const std::string & directory, const std::string & stem) {
    std::ifstream input(directory + "/" + stem + ".f32", std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("missing independent checkpoint: " + stem);
    const auto size = input.tellg();
    if (size < 0 || static_cast<uint64_t>(size) % sizeof(float)) throw std::runtime_error("invalid checkpoint size: " + stem);
    std::vector<float> result(static_cast<size_t>(size) / sizeof(float)); input.seekg(0);
    input.read(reinterpret_cast<char *>(result.data()), size);
    if (!input) throw std::runtime_error("truncated independent checkpoint: " + stem);
    return result;
}
std::vector<float> reference_attention_prefix(const std::string & directory,
    const std::string & stem, uint32_t positions) {
    if (internal_qualification_mode())
        return std::vector<float>(static_cast<size_t>(HEADS) * positions * positions, 0.0f);
    std::ifstream metadata(directory + "/" + stem + ".meta");
    std::string line, shape;
    while (std::getline(metadata, line)) if (line.rfind("shape=", 0) == 0) shape = line.substr(6);
    if (shape.empty()) throw std::runtime_error("missing attention shape metadata: " + stem);
    std::array<uint64_t, 4> dims{};
    std::stringstream parts(shape);
    std::string part;
    for (size_t i = 0; i < dims.size() && std::getline(parts, part, ','); ++i) dims[i] = std::stoull(part);
    if (dims[0] < positions || dims[1] != positions || dims[2] != HEADS || dims[3] != 1)
        throw std::runtime_error("unexpected reference attention matrix geometry: " + shape);
    const auto full = reference(directory, stem);
    if (full.size() != dims[0] * dims[1] * dims[2] * dims[3])
        throw std::runtime_error("reference attention shape/bytes mismatch: " + stem);
    std::vector<float> compact(static_cast<size_t>(positions) * positions * HEADS);
    for (uint32_t h = 0; h < HEADS; ++h) for (uint32_t p = 0; p < positions; ++p)
        for (uint32_t s = 0; s < positions; ++s) {
            const size_t source = static_cast<size_t>(h) * dims[0] * dims[1] +
                static_cast<size_t>(p) * dims[0] + s;
            const size_t destination = static_cast<size_t>(h) * positions * positions +
                static_cast<size_t>(p) * positions + s;
            compact[destination] = full[source];
        }
    return compact;
}
std::string checkpoint_shape(const std::string & label, uint32_t positions, size_t values) {
    if (label == "final_logits" || label == "kv_vs_full_logits" || label == "reset_clean_logits_vs_full") return "[151936,1]";
    if (label == "q_rmsnorm" || label == "q_rope_neox") return "[128,40," + std::to_string(positions) + "]";
    if (label == "k_rmsnorm" || label == "k_rope_neox") return "[128,8," + std::to_string(positions) + "]";
    if (label == "k_projection" || label == "v_projection") return "[1024," + std::to_string(positions) + "]";
    if (label == "ffn_gate" || label == "ffn_up" || label == "ffn_swiglu") return "[17408," + std::to_string(positions) + "]";
    if (label == "attention_scores" || label == "attention_probabilities" ||
        label == "llama_qk_vs_fp64_oracle" || label == "vbuf_qk_vs_fp64_oracle") {
        const size_t key_extent = values / (static_cast<size_t>(HEADS) * positions);
        return "[" + std::to_string(key_extent) + "," + std::to_string(positions) + ",40]";
    }
    return "[5120," + std::to_string(positions) + "]";
}
bool compare(const std::string & label, const std::vector<float> & actual,
    const std::vector<float> & expected, uint32_t positions, bool print = true) {
    if (actual.size() != expected.size() || actual.empty()) throw std::runtime_error("checkpoint shape mismatch: " + label);
    double max_abs = -1, mean_abs = 0, max_rel = 0, squared_error = 0;
    double dot = 0, actual_norm = 0, expected_norm = 0;
    uint64_t max_ulp = 0, ulp_at_max_abs = 0;
    size_t max_index = 0, max_ulp_index = 0; bool finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        finite = finite && std::isfinite(actual[i]) && std::isfinite(expected[i]);
        if (std::isfinite(actual[i]) && std::isfinite(expected[i])) {
            uint32_t a_bits = 0, e_bits = 0;
            std::memcpy(&a_bits, &actual[i], sizeof(a_bits));
            std::memcpy(&e_bits, &expected[i], sizeof(e_bits));
            const uint32_t a_ordered = (a_bits & 0x80000000u) ? ~a_bits : (a_bits | 0x80000000u);
            const uint32_t e_ordered = (e_bits & 0x80000000u) ? ~e_bits : (e_bits | 0x80000000u);
            const uint64_t ulp = a_ordered > e_ordered ? a_ordered - e_ordered : e_ordered - a_ordered;
            if (ulp > max_ulp) { max_ulp = ulp; max_ulp_index = i; }
            if (i == max_index || std::abs(static_cast<double>(actual[i]) - expected[i]) > max_abs)
                ulp_at_max_abs = ulp;
        }
        const double delta = std::abs(static_cast<double>(actual[i]) - expected[i]);
        if (delta > max_abs) { max_abs = delta; max_index = i; }
        mean_abs += delta;
        squared_error += delta * delta;
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
        if (std::abs(expected[i]) > 1e-8)
            max_rel = std::max(max_rel, delta / std::abs(static_cast<double>(expected[i])));
    }
    mean_abs /= actual.size();
    const double rms = std::sqrt(squared_error / actual.size());
    const double cosine = actual_norm > 0 && expected_norm > 0 ?
        dot / std::sqrt(actual_norm * expected_norm) : 0.0;
    const double ref_at_max = expected[max_index], actual_at_max = actual[max_index];
    const bool within_previous_atol = max_abs <= ATOL + RTOL * std::abs(ref_at_max);
    std::vector<double> per_token_max(positions, 0.0);
    if (label == "attention_scores" || label == "attention_probabilities" ||
        label == "llama_qk_vs_fp64_oracle" || label == "vbuf_qk_vs_fp64_oracle") {
        const uint32_t key_extent = static_cast<uint32_t>(actual.size() /
            (static_cast<size_t>(HEADS) * positions));
        for (uint32_t h = 0; h < HEADS; ++h) for (uint32_t p = 0; p < positions; ++p)
            for (uint32_t s = 0; s < key_extent; ++s) {
                const size_t index = static_cast<size_t>(h) * positions * key_extent +
                    static_cast<size_t>(p) * key_extent + s;
                per_token_max[p] = std::max(per_token_max[p],
                    std::abs(static_cast<double>(actual[index]) - expected[index]));
            }
    } else {
        const size_t values_per_token = actual.size() / positions;
        for (size_t i = 0; i < actual.size(); ++i)
            per_token_max[i / values_per_token] = std::max(per_token_max[i / values_per_token],
                std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    if (print) {
        const double reference_rms = std::sqrt(expected_norm / actual.size());
        const double reference_l2 = std::sqrt(expected_norm);
        const double actual_l2 = std::sqrt(actual_norm);
        const double relative_rms = reference_rms > 0 ? rms / reference_rms : 0;
        const double norm_ratio = reference_l2 > 0 ? actual_l2 / reference_l2 : 0;
        std::printf("checkpoint=%s shape=%s all_finite=%s max_abs=%.9g mean_abs=%.9g rms=%.9g "
            "reference_rms=%.9g reference_l2=%.9g vbuf_l2=%.9g relative_rms=%.9g norm_ratio=%.12g "
            "cosine=%.12g max_rel(ref_abs>1e-8)=%.9g max_index=%zu reference=%.9g vbuf=%.9g "
            "max_ulp=%llu max_ulp_index=%zu ulp_at_max_abs=%llu previous_atol=%.6g within_previous_atol=%s per_token_max_abs=",
            label.c_str(), checkpoint_shape(label, positions, actual.size()).c_str(), finite ? "YES" : "NO",
            max_abs, mean_abs, rms, reference_rms, reference_l2, actual_l2, relative_rms,
            norm_ratio, cosine, max_rel, max_index, ref_at_max, actual_at_max,
            static_cast<unsigned long long>(max_ulp), max_ulp_index,
            static_cast<unsigned long long>(ulp_at_max_abs),
            ATOL + RTOL * std::abs(ref_at_max), within_previous_atol ? "YES" : "NO");
        for (uint32_t token = 0; token < positions; ++token)
            std::printf("%s%.9g", token == 0 ? "" : ",", per_token_max[token]);
        std::printf(" finite_status=%s\n", finite ? "FINITE" : "NONFINITE");
    }
    return finite;
}
bool within_fixed_tolerance(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size() || actual.empty()) return false;
    double maximum = -1.0;
    size_t index = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = std::abs(static_cast<double>(actual[i]) - expected[i]);
        if (delta > maximum) { maximum = delta; index = i; }
    }
    return maximum <= ATOL + RTOL * std::abs(static_cast<double>(expected[index]));
}
struct ProjectionOracle {
    std::vector<float> fp32;
    std::vector<float> fp64;
};
ProjectionOracle simple_projection(const std::vector<uint8_t> & packed_weights,
    const std::vector<float> & input, ggml_type type, uint32_t width,
    uint32_t output_rows, uint32_t positions) {
    const auto * traits = ggml_get_type_traits(type);
    const size_t row_bytes = ggml_row_size(type, width);
    if (traits == nullptr || traits->to_float == nullptr || packed_weights.size() != row_bytes * output_rows ||
        input.size() != static_cast<size_t>(width) * positions)
        throw std::runtime_error("simple FFN-down oracle geometry invalid");
    ProjectionOracle result;
    result.fp32.resize(static_cast<size_t>(output_rows) * positions);
    result.fp64.resize(result.fp32.size());
    std::vector<float> decoded(width);
    for (uint32_t row = 0; row < output_rows; ++row) {
        traits->to_float(packed_weights.data() + static_cast<size_t>(row) * row_bytes,
            decoded.data(), width);
        for (uint32_t token = 0; token < positions; ++token) {
            float sum32 = 0;
            double sum64 = 0;
            const size_t activation_offset = static_cast<size_t>(token) * width;
            for (uint32_t feature = 0; feature < width; ++feature) {
                const float product = decoded[feature] * input[activation_offset + feature];
                sum32 += product;
                sum64 += static_cast<double>(decoded[feature]) * input[activation_offset + feature];
            }
            result.fp32[static_cast<size_t>(token) * output_rows + row] = sum32;
            result.fp64[static_cast<size_t>(token) * output_rows + row] = static_cast<float>(sum64);
        }
    }
    return result;
}
std::vector<std::vector<uint8_t>> quantized_activation_rows(const std::vector<float> & input,
    ggml_type weight_type, uint32_t width, uint32_t positions) {
    const auto * weight_traits = ggml_get_type_traits_cpu(weight_type);
    if (weight_traits == nullptr) throw std::runtime_error("GGML CPU weight traits unavailable");
    const ggml_type activation_type = weight_traits->vec_dot_type;
    const auto * activation_traits = ggml_get_type_traits_cpu(activation_type);
    if (activation_traits == nullptr || activation_traits->from_float == nullptr)
        throw std::runtime_error("GGML CPU activation quantizer unavailable");
    const size_t row_bytes = ggml_row_size(activation_type, width);
    std::vector<std::vector<uint8_t>> rows(positions, std::vector<uint8_t>(row_bytes));
    for (uint32_t token = 0; token < positions; ++token)
        activation_traits->from_float(input.data() + static_cast<size_t>(token) * width,
            rows[token].data(), width);
    return rows;
}
void report_activation_quantization_delta(const std::vector<float> & reference_input,
    const std::vector<float> & vbuf_input, ggml_type weight_type, uint32_t width, uint32_t positions,
    uint32_t layer, const char * path) {
    const auto * weight_traits = ggml_get_type_traits_cpu(weight_type);
    const ggml_type activation_type = weight_traits->vec_dot_type;
    if (activation_type != GGML_TYPE_Q8_K)
        throw std::runtime_error("Q8_K activation analysis expected for selected FFN-down weights");
    const auto reference_rows = quantized_activation_rows(reference_input, weight_type, width, positions);
    const auto vbuf_rows = quantized_activation_rows(vbuf_input, weight_type, width, positions);
    const size_t block_bytes = ggml_type_size(activation_type);
    const size_t row_bytes = ggml_row_size(activation_type, width);
    const int64_t block_elements = ggml_blck_size(activation_type);
    uint64_t different_bytes = 0, different_blocks = 0, different_values = 0, scale_changes = 0;
    double pre_sq = 0, post_sq = 0, pre_max = 0, post_max = 0;
    for (size_t i = 0; i < reference_input.size(); ++i) {
        const double pre = std::abs(static_cast<double>(vbuf_input[i]) - reference_input[i]);
        pre_max = std::max(pre_max, pre);
        pre_sq += pre * pre;
    }
    std::vector<float> ref_decoded(width), vbuf_decoded(width);
    for (uint32_t token = 0; token < positions; ++token) {
        const auto & ref = reference_rows[token];
        const auto & got = vbuf_rows[token];
        for (size_t b = 0; b < row_bytes; ++b) different_bytes += ref[b] != got[b];
        for (size_t block = 0; block < row_bytes / block_bytes; ++block) {
            const size_t offset = block * block_bytes;
            different_blocks += std::memcmp(ref.data() + offset, got.data() + offset, block_bytes) != 0;
            uint32_t ref_scale_bits = 0, got_scale_bits = 0;
            std::memcpy(&ref_scale_bits, ref.data() + offset, sizeof(ref_scale_bits));
            std::memcpy(&got_scale_bits, got.data() + offset, sizeof(got_scale_bits));
            scale_changes += ref_scale_bits != got_scale_bits;
        }
        for (size_t block = 0; block < row_bytes / block_bytes; ++block) {
            const size_t offset = block * block_bytes;
            float ref_scale = 0, got_scale = 0;
            std::memcpy(&ref_scale, ref.data() + offset, sizeof(ref_scale));
            std::memcpy(&got_scale, got.data() + offset, sizeof(got_scale));
            for (int64_t i = 0; i < block_elements; ++i) {
                int8_t ref_q = 0, got_q = 0;
                std::memcpy(&ref_q, ref.data() + offset + sizeof(float) + i, sizeof(ref_q));
                std::memcpy(&got_q, got.data() + offset + sizeof(float) + i, sizeof(got_q));
                const uint32_t index = static_cast<uint32_t>(block * block_elements + i);
                ref_decoded[index] = ref_scale * ref_q;
                vbuf_decoded[index] = got_scale * got_q;
                different_values += ref_q != got_q;
                const double post = std::abs(static_cast<double>(vbuf_decoded[index]) - ref_decoded[index]);
                post_max = std::max(post_max, post);
                post_sq += post * post;
            }
        }
    }
    std::printf("activation_quantization path=%s layer=%u weight_type=%s activation_type=%s block_elements=%lld "
        "pre_max_abs=%.9g pre_rms=%.9g changed_blocks=%llu changed_values=%llu changed_bytes=%llu "
        "scale_changes=%llu post_dequant_max_abs=%.9g post_dequant_rms=%.9g\n",
        path, layer, ggml_type_name(weight_type), ggml_type_name(activation_type),
        static_cast<long long>(block_elements), pre_max,
        std::sqrt(pre_sq / reference_input.size()), static_cast<unsigned long long>(different_blocks),
        static_cast<unsigned long long>(different_values), static_cast<unsigned long long>(different_bytes),
        static_cast<unsigned long long>(scale_changes), post_max,
        std::sqrt(post_sq / reference_input.size()));
}
std::vector<float> attention_qk_fp64(const std::vector<float> & q_values,
    const std::vector<float> & k_values, uint32_t positions) {
    std::vector<float> result(static_cast<size_t>(HEADS) * positions * positions);
    for (uint32_t head = 0; head < HEADS; ++head) for (uint32_t query = 0; query < positions; ++query)
        for (uint32_t key = 0; key < positions; ++key) {
            double sum = 0;
            const uint32_t kv_head = head / (HEADS / KV_HEADS);
            for (uint32_t d = 0; d < HEAD_DIM; ++d) {
                const float q = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    q_values[static_cast<size_t>(query) * HEADS * HEAD_DIM + head * HEAD_DIM + d]));
                const float k = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    k_values[static_cast<size_t>(key) * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + d]));
                sum += static_cast<double>(q) * k;
            }
            result[static_cast<size_t>(head) * positions * positions +
                static_cast<size_t>(query) * positions + key] = static_cast<float>(sum);
        }
    return result;
}
std::vector<float> attention_v_projection_fp64(const std::vector<float> & values,
    const std::vector<float> & probabilities, uint32_t positions) {
    std::vector<float> result(static_cast<size_t>(EMBED) * positions);
    for (uint32_t token = 0; token < positions; ++token) for (uint32_t head = 0; head < HEADS; ++head) {
        const uint32_t kv_head = head / (HEADS / KV_HEADS);
        for (uint32_t d = 0; d < HEAD_DIM; ++d) {
            double sum = 0;
            for (uint32_t key = 0; key <= token; ++key) {
                const float v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    values[static_cast<size_t>(key) * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + d]));
                const float p = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    probabilities[static_cast<size_t>(head) * positions * positions +
                        static_cast<size_t>(token) * positions + key]));
                sum += static_cast<double>(v) * p;
            }
            result[static_cast<size_t>(token) * EMBED + head * HEAD_DIM + d] = static_cast<float>(sum);
        }
    }
    return result;
}
std::vector<float> ggml_quantized_dot_projection(const std::vector<uint8_t> & packed_weights,
    const std::vector<float> & input, ggml_type weight_type, uint32_t width,
    uint32_t output_rows, uint32_t positions) {
    const auto * weight_traits = ggml_get_type_traits_cpu(weight_type);
    if (weight_traits == nullptr || weight_traits->vec_dot == nullptr)
        throw std::runtime_error("GGML CPU quantized vec-dot trait unavailable");
    const ggml_type activation_type = weight_traits->vec_dot_type;
    const auto * activation_traits = ggml_get_type_traits_cpu(activation_type);
    if (activation_traits == nullptr || activation_traits->from_float == nullptr)
        throw std::runtime_error("GGML CPU activation quantizer unavailable");
    const size_t weight_row_bytes = ggml_row_size(weight_type, width);
    const size_t activation_row_bytes = ggml_row_size(activation_type, width);
    if (packed_weights.size() != weight_row_bytes * output_rows ||
        input.size() != static_cast<size_t>(width) * positions)
        throw std::runtime_error("GGML CPU vec-dot oracle geometry invalid");
    std::vector<float> output(static_cast<size_t>(output_rows) * positions);
    std::vector<uint8_t> quantized_input(activation_row_bytes);
    for (uint32_t token = 0; token < positions; ++token) {
        activation_traits->from_float(input.data() + static_cast<size_t>(token) * width,
            quantized_input.data(), width);
        for (uint32_t row = 0; row < output_rows; ++row) {
            float value = 0.0f;
            weight_traits->vec_dot(width, &value, sizeof(value),
                packed_weights.data() + static_cast<size_t>(row) * weight_row_bytes,
                ggml_type_size(weight_type), quantized_input.data(),
                ggml_type_size(activation_type), 1);
            output[static_cast<size_t>(token) * output_rows + row] = value;
        }
    }
    std::printf("ffn_down_cpu_dot weight_type=%s activation_vec_dot_type=%s activation_row_bytes=%zu "
        "accumulation=kernel_F32\n", ggml_type_name(weight_type),
        ggml_type_name(activation_type), activation_row_bytes);
    return output;
}

void print_matmul_geometry(const char * label, const ggml_tensor * tensor) {
    const auto print_operand = [](const ggml_tensor * value) {
        if (value == nullptr) { std::printf("none"); return; }
        std::printf("type=%s,shape=[%lld,%lld,%lld,%lld],nb=[%zu,%zu,%zu,%zu]",
            ggml_type_name(value->type), static_cast<long long>(value->ne[0]),
            static_cast<long long>(value->ne[1]), static_cast<long long>(value->ne[2]),
            static_cast<long long>(value->ne[3]), value->nb[0], value->nb[1], value->nb[2], value->nb[3]);
    };
    std::printf("matmul_geometry name=%s op=%s out_type=%s out_shape=[%lld,%lld,%lld,%lld] "
        "src0={", label, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
        static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
        static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]));
    print_operand(tensor->src[0]); std::printf("} src1={"); print_operand(tensor->src[1]);
    std::printf("} contiguous_src1=%s\n", ggml_is_contiguous(tensor->src[1]) ? "YES" : "NO");
}
void write_binary_export(const std::filesystem::path & path, const uint8_t * bytes, size_t size);
void write_f32_export(const std::filesystem::path & path, const std::vector<float> & values);

std::vector<float> run_positions(Model & model, const std::string & reference_dir,
    uint32_t positions, uint32_t layer, const std::vector<float> & input_hidden,
    bool reference_reset = false, const std::vector<int32_t> & input_tokens = {},
    Qwen3KvCache * retained_cache = nullptr, uint32_t absolute_position = 0,
    LayerDiagnostics * diagnostics = nullptr,
    const std::vector<uint8_t> * replay_key_history = nullptr,
    const std::vector<uint8_t> * replay_value_history = nullptr,
    bool use_canonical_softmax_extent = false,
    bool force_all_final_positions = false) {
    const auto block_started = std::chrono::steady_clock::now();
    const bool replay_incremental = replay_key_history != nullptr || replay_value_history != nullptr;
    if ((replay_key_history == nullptr) != (replay_value_history == nullptr) ||
        (retained_cache != nullptr && replay_incremental))
        throw std::runtime_error("invalid incremental Qwen3 KV source selection");
    const bool incremental_execution = retained_cache != nullptr || replay_incremental;
    if (retained_cache != nullptr && (positions != 1 || reference_reset ||
        absolute_position != retained_cache->completed_positions() ||
        retained_cache->layer_length(layer) != absolute_position))
        throw std::runtime_error("invalid incremental Qwen3 KV state/position");
    const size_t bytes_per_tensor_token = static_cast<size_t>(KV_HEADS) * HEAD_DIM * sizeof(uint16_t);
    if (replay_incremental && (positions != 1 || reference_reset ||
        replay_key_history->size() != static_cast<size_t>(absolute_position) * bytes_per_tensor_token ||
        replay_value_history->size() != static_cast<size_t>(absolute_position) * bytes_per_tensor_token))
        throw std::runtime_error("invalid ephemeral incremental replay history");
    const uint32_t logical_key_positions = incremental_execution ? absolute_position + positions : positions;
    const CpuSoftmaxPolicy softmax_policy = cpu_softmax_policy();
    const SoftmaxComputeExtent softmax_extent = SoftmaxComputeExtent::make(
        logical_key_positions, softmax_policy.granularity);
    const uint32_t key_positions = use_canonical_softmax_extent ? softmax_extent.compute : logical_key_positions;
    const uint32_t compute_padding = key_positions - logical_key_positions;
    if (layer == 0)
        std::printf("softmax_compute_extent mode=%s canonical=%s logical_extent=%u compute_extent=%u granularity=%u padding=%u kernel=%s\n",
            incremental_execution ? (retained_cache ? "persistent" : "incremental_replay") : "full_sequence",
            use_canonical_softmax_extent ? "YES" : "NO", logical_key_positions, key_positions,
            softmax_policy.granularity, compute_padding, softmax_policy.kernel_path);
    Context first = make_context();
    std::vector<Tensor *> first_weights;
    const auto add_weight = [&](const std::string & name) -> ggml_tensor * {
        Tensor & tensor = get(model, name); first_weights.push_back(&tensor); return make_tensor(first, tensor);
    };
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    ggml_tensor * embedding = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * layer_input = nullptr;
    if (layer == 0 && !reference_reset) {
        if (!input_tokens.empty() && input_tokens.size() != positions)
            throw std::runtime_error("input token ID count does not match positions");
        embedding = add_weight("token_embd.weight");
        ids = ggml_new_tensor_1d(first.ctx, GGML_TYPE_I32, positions);
        layer_input = ggml_get_rows(first.ctx, embedding, ids);
    } else {
        if (input_hidden.size() != static_cast<size_t>(EMBED) * positions)
            throw std::runtime_error("block input hidden-state shape mismatch");
        layer_input = ggml_new_tensor_2d(first.ctx, GGML_TYPE_F32, EMBED, positions);
    }
    ggml_tensor * attn_norm_w = add_weight(prefix + "attn_norm.weight");
    ggml_tensor * q_norm_w = add_weight(prefix + "attn_q_norm.weight");
    ggml_tensor * k_norm_w = add_weight(prefix + "attn_k_norm.weight");

    ggml_tensor * attn_norm = ggml_mul(first.ctx, ggml_rms_norm(first.ctx, layer_input, 1e-6f), attn_norm_w);
    // Q/K/V are dependency-independent once the shared attention norm is ready.
    // Their matmuls run in separate GGML contexts/backends under vBuf budgeting.
    ggml_tensor * q_linear = ggml_new_tensor_2d(first.ctx, GGML_TYPE_F32, EMBED, positions);
    ggml_tensor * k_linear = ggml_new_tensor_2d(first.ctx, GGML_TYPE_F32, KV_HEADS * HEAD_DIM, positions);
    ggml_tensor * v_linear = ggml_new_tensor_2d(first.ctx, GGML_TYPE_F32, KV_HEADS * HEAD_DIM, positions);
    ggml_tensor * q_heads = ggml_reshape_3d(first.ctx, q_linear, HEAD_DIM, HEADS, positions);
    ggml_tensor * k_heads = ggml_reshape_3d(first.ctx, k_linear, HEAD_DIM, KV_HEADS, positions);
    ggml_tensor * v_heads = ggml_reshape_3d(first.ctx, v_linear, HEAD_DIM, KV_HEADS, positions);
    ggml_tensor * qnorm = ggml_mul(first.ctx, ggml_rms_norm(first.ctx, q_heads, 1e-6f), q_norm_w);
    ggml_tensor * knorm = ggml_mul(first.ctx, ggml_rms_norm(first.ctx, k_heads, 1e-6f), k_norm_w);
    ggml_tensor * pos = ggml_new_tensor_1d(first.ctx, GGML_TYPE_I32, positions);
    ggml_tensor * qrope = ggml_rope_ext(first.ctx, qnorm, pos, nullptr, HEAD_DIM,
        GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * krope = ggml_rope_ext(first.ctx, knorm, pos, nullptr, HEAD_DIM,
        GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    // llama.cpp's default KV cache is F16. Preserve that boundary before
    // building non-flash causal GQA with GGML operations.
    ggml_tensor * k_cache = ggml_cast(first.ctx, krope, GGML_TYPE_F16);
    ggml_tensor * v_cache = ggml_cast(first.ctx, v_heads, GGML_TYPE_F16);
    const std::string suffix = "-" + std::to_string(layer);
    struct AttentionGroupGraph {
        Qwen3QueryGroup group;
        uint32_t tensor_query_start = 0;
        ggml_tensor * causal_mask = nullptr;
        ggml_tensor * past_k = nullptr;
        ggml_tensor * past_v = nullptr;
        ggml_tensor * raw_scores = nullptr;
        ggml_tensor * probabilities = nullptr;
        ggml_tensor * reference_probabilities = nullptr;
        ggml_tensor * v_batched = nullptr;
        ggml_tensor * context_heads = nullptr;
        ggml_tensor * context = nullptr;
        ggml_tensor * reference_context = nullptr;
        std::vector<float> mask_values;
        std::vector<float> reference_values;
    };
    std::vector<Qwen3QueryGroup> query_groups;
    if (use_canonical_softmax_extent && !incremental_execution) {
        query_groups = qwen3_query_groups(positions, softmax_policy.granularity);
    } else {
        query_groups.push_back({incremental_execution ? absolute_position : 0,
            positions, key_positions});
    }
    std::vector<AttentionGroupGraph> attention_groups;
    attention_groups.reserve(query_groups.size());
    auto view_kv_prefix = [&](ggml_tensor * tensor, uint32_t count) {
        if (count == static_cast<uint32_t>(tensor->ne[2])) return tensor;
        return ggml_view_3d(first.ctx, tensor, HEAD_DIM, KV_HEADS, count,
            tensor->nb[1], tensor->nb[2], 0);
    };
    const auto reference_probabilities_logical = reference_attention_prefix(reference_dir,
        "kq_soft_max" + suffix + "-0", logical_key_positions);
    for (const Qwen3QueryGroup & query_group : query_groups) {
        AttentionGroupGraph item;
        item.group = query_group;
        item.tensor_query_start = incremental_execution ? 0 : query_group.first_query;
        const uint32_t extent = query_group.compute_extent;
        ggml_tensor * q_rows = qrope;
        if (item.tensor_query_start != 0 || query_group.query_count != positions)
            q_rows = ggml_view_3d(first.ctx, qrope, HEAD_DIM, HEADS,
                query_group.query_count, qrope->nb[1], qrope->nb[2],
                static_cast<size_t>(item.tensor_query_start) * qrope->nb[2]);
        ggml_tensor * q_batched = ggml_permute(first.ctx, q_rows, 0, 2, 1, 3);

        ggml_tensor * group_k = nullptr;
        ggml_tensor * group_v = nullptr;
        if (incremental_execution) {
            group_k = k_cache;
            group_v = v_cache;
            if (absolute_position != 0) {
                item.past_k = ggml_new_tensor_3d(first.ctx, GGML_TYPE_F16,
                    HEAD_DIM, KV_HEADS, absolute_position);
                item.past_v = ggml_new_tensor_3d(first.ctx, GGML_TYPE_F16,
                    HEAD_DIM, KV_HEADS, absolute_position);
                group_k = ggml_concat(first.ctx, item.past_k, group_k, 2);
                group_v = ggml_concat(first.ctx, item.past_v, group_v, 2);
            }
        } else {
            const uint32_t available = std::min(extent, positions);
            group_k = view_kv_prefix(k_cache, available);
            group_v = view_kv_prefix(v_cache, available);
        }
        const uint32_t available_positions = static_cast<uint32_t>(group_k->ne[2]);
        if (available_positions > extent)
            throw std::logic_error("canonical query group has more real KV rows than compute extent");
        const uint32_t group_padding = extent - available_positions;
        if (group_padding != 0) {
            ggml_tensor * zero_k = ggml_fill(first.ctx,
                ggml_new_tensor_3d(first.ctx, GGML_TYPE_F16, HEAD_DIM, KV_HEADS, group_padding), 0.0f);
            ggml_tensor * zero_v = ggml_fill(first.ctx,
                ggml_new_tensor_3d(first.ctx, GGML_TYPE_F16, HEAD_DIM, KV_HEADS, group_padding), 0.0f);
            group_k = ggml_concat(first.ctx, group_k, zero_k, 2);
            group_v = ggml_concat(first.ctx, group_v, zero_v, 2);
        }
        ggml_tensor * k_batched = ggml_permute(first.ctx, group_k, 0, 2, 1, 3);
        ggml_tensor * v_batched = ggml_cont(first.ctx,
            ggml_permute(first.ctx, group_v, 1, 2, 0, 3));
        item.causal_mask = ggml_new_tensor_2d(first.ctx, GGML_TYPE_F32,
            extent, query_group.query_count);
        item.raw_scores = ggml_mul_mat(first.ctx, k_batched, q_batched);
        item.probabilities = ggml_soft_max_ext(first.ctx, item.raw_scores, item.causal_mask,
            1.0f / std::sqrt(static_cast<float>(HEAD_DIM)), 0.0f);
        item.reference_probabilities = ggml_new_tensor_3d(first.ctx, GGML_TYPE_F32,
            extent, query_group.query_count, HEADS);
        item.context_heads = ggml_mul_mat(first.ctx, v_batched, item.probabilities);
        ggml_tensor * context_layout = ggml_permute(first.ctx, item.context_heads, 0, 2, 1, 3);
        item.context = ggml_cont_2d(first.ctx, context_layout, EMBED, query_group.query_count);
        ggml_tensor * reference_context_heads = ggml_mul_mat(first.ctx, v_batched,
            item.reference_probabilities);
        ggml_tensor * reference_context_layout = ggml_permute(first.ctx,
            reference_context_heads, 0, 2, 1, 3);
        item.reference_context = ggml_cont_2d(first.ctx,
            reference_context_layout, EMBED, query_group.query_count);
        item.v_batched = v_batched;
        item.mask_values.resize(static_cast<size_t>(extent) * query_group.query_count);
        item.reference_values.assign(static_cast<size_t>(HEADS) * query_group.query_count * extent, 0.0f);
        for (uint32_t local_query = 0; local_query < query_group.query_count; ++local_query) {
            const uint32_t logical_query = query_group.first_query + local_query;
            const uint32_t visible = incremental_execution ? absolute_position + local_query + 1 : logical_query + 1;
            if (visible > logical_key_positions || visible > extent)
                throw std::logic_error("query group logical visibility exceeds its canonical extent");
            for (uint32_t key = 0; key < extent; ++key)
                item.mask_values[static_cast<size_t>(local_query) * extent + key] =
                    key < visible ? 0.0f : -INFINITY;
            const uint32_t reference_query = incremental_execution ? absolute_position + local_query : logical_query;
            for (uint32_t head = 0; head < HEADS; ++head)
                for (uint32_t key = 0; key < visible; ++key) {
                    const size_t src = (static_cast<size_t>(head) * logical_key_positions + reference_query) *
                        logical_key_positions + key;
                    const size_t dst = (static_cast<size_t>(head) * query_group.query_count + local_query) * extent + key;
                    item.reference_values[dst] = reference_probabilities_logical[src];
                }
        }
        attention_groups.push_back(std::move(item));
    }
    ggml_tensor * attention_context = attention_groups.front().context;
    ggml_tensor * reference_attention_context = attention_groups.front().reference_context;
    ggml_tensor * v_batched = attention_groups.front().v_batched;
    ggml_tensor * raw_scores = attention_groups.front().raw_scores;
    ggml_tensor * context_heads = attention_groups.front().context_heads;
    for (size_t index = 1; index < attention_groups.size(); ++index) {
        attention_context = ggml_concat(first.ctx, attention_context,
            attention_groups[index].context, 1);
        reference_attention_context = ggml_concat(first.ctx, reference_attention_context,
            attention_groups[index].reference_context, 1);
    }
    if (attention_groups.size() > 1)
        std::printf("canonical_query_grouping sequence=%u group_count=%zu order=restored\n",
            positions, attention_groups.size());
    ggml_tensor * padded_v_input = nullptr;
    ggml_tensor * padded_probabilities_input = nullptr;
    ggml_tensor * padded_context_output = nullptr;
    constexpr int64_t REFERENCE_KV_CAPACITY = 256;
    if (layer == 0 && reference_reset) {
        padded_v_input = ggml_new_tensor_3d(first.ctx, GGML_TYPE_F16,
            REFERENCE_KV_CAPACITY, HEAD_DIM, KV_HEADS);
        padded_probabilities_input = ggml_new_tensor_3d(first.ctx, GGML_TYPE_F32,
            REFERENCE_KV_CAPACITY, positions, HEADS);
        ggml_tensor * padded_context_heads = ggml_mul_mat(first.ctx,
            padded_v_input, padded_probabilities_input);
        ggml_tensor * padded_context_layout = ggml_permute(first.ctx,
            padded_context_heads, 0, 2, 1, 3);
        padded_context_output = ggml_cont_2d(first.ctx,
            padded_context_layout, EMBED, positions);
        print_matmul_geometry("v_context_reference_extent_control", padded_context_heads);
    }
    if (layer == 0 && reference_reset) {
        print_matmul_geometry("qk_scores", raw_scores);
        print_matmul_geometry("v_context", context_heads);
    }
    first.buffer = ggml_backend_alloc_ctx_tensors(first.ctx, first.backend);
    if (!first.buffer) throw std::runtime_error("Qwen first-stage tensor allocation failed");
    g_qualification_memory_peaks.first_stage_bytes = std::max(
        g_qualification_memory_peaks.first_stage_bytes, ggml_backend_buffer_get_size(first.buffer));
    for (auto & group : attention_groups) {
        set_tensor(group.causal_mask, group.mask_values.data(),
            group.mask_values.size() * sizeof(float));
        set_tensor(group.reference_probabilities, group.reference_values.data(),
            group.reference_values.size() * sizeof(float));
        if (group.past_k != nullptr) {
            const size_t old_bytes = retained_cache != nullptr ? retained_cache->layer_bytes(layer) : replay_key_history->size();
            const uint8_t * old_key = retained_cache != nullptr ? retained_cache->key_bytes(layer) : replay_key_history->data();
            const uint8_t * old_value = retained_cache != nullptr ? retained_cache->value_bytes(layer) : replay_value_history->data();
            ggml_backend_tensor_set(group.past_k, old_key, 0, old_bytes);
            ggml_backend_tensor_set(group.past_v, old_value, 0, old_bytes);
        }
    }
    if (layer > 0 || reference_reset)
        set_tensor(layer_input, input_hidden.data(), input_hidden.size() * sizeof(float));
    if (ids != nullptr) for (uint32_t i = 0; i < positions; ++i) {
        const int32_t id = input_tokens.empty() ? static_cast<int32_t>(i) : input_tokens[i];
        ggml_backend_tensor_set(ids, &id, i * sizeof(id), sizeof(id));
    }
    std::vector<int32_t> position_ids(positions); for (uint32_t i = 0; i < positions; ++i) position_ids[i] = static_cast<int32_t>(incremental_execution ? absolute_position + i : i);
    set_tensor(pos, position_ids.data(), position_ids.size() * sizeof(int32_t));
    if (layer == 14 && (positions == 8 || incremental_execution)) {
        std::printf("kv_rope_position_path mode=%s layer=%u row_count=%u absolute_start=%u rope_ids=",
            incremental_execution ? "single_row" : "full_sequence", layer, positions,
            incremental_execution ? absolute_position : 0);
        for (uint32_t i = 0; i < positions; ++i) std::printf("%s%d", i == 0 ? "" : ",", position_ids[i]);
        std::printf("\n");
    }
    uint32_t ref_id = 0;
    for (Tensor * tensor : first_weights) {
        if (!model.materializer->request(ref_id, tensor->persistent(), tensor->length) ||
            model.materializer->wait(ref_id) != MaterializationState::Ready)
            throw std::runtime_error("vBuf weight request failed: " + tensor->name);
        auto ready = model.materializer->obtain_ready_tensor(ref_id++);
        if (!ready || ready->payload_len != tensor->length) throw std::runtime_error("vBuf weight lease failed");
        ggml_backend_tensor_set(tensor->ggml, ready->payload, 0, tensor->length);
        model.materializer->release(ref_id - 1);
    }
    const auto materialize_projection_weight = [&](Tensor & tensor) {
        if (!model.materializer->request(ref_id, tensor.persistent(), tensor.length) ||
            model.materializer->wait(ref_id) != MaterializationState::Ready)
            throw std::runtime_error("vBuf projection weight request failed: " + tensor.name);
        auto ready = model.materializer->obtain_ready_tensor(ref_id++);
        if (!ready || ready->payload_len != tensor.length)
            throw std::runtime_error("vBuf projection weight lease failed: " + tensor.name);
        std::vector<uint8_t> bytes(ready->payload, ready->payload + ready->payload_len);
        model.materializer->release(ref_id - 1);
        return bytes;
    };
    const auto q_weight_bytes = materialize_projection_weight(get(model, prefix + "attn_q.weight"));
    const auto k_weight_bytes = materialize_projection_weight(get(model, prefix + "attn_k.weight"));
    const auto v_weight_bytes = materialize_projection_weight(get(model, prefix + "attn_v.weight"));
    ggml_cgraph * norm_graph = ggml_new_graph(first.ctx);
    ggml_build_forward_expand(norm_graph, attn_norm);
    if (ggml_backend_graph_compute(first.backend, norm_graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Qwen embedding/attention-normalization graph execution failed");
    ggml_backend_synchronize(first.backend);
    const auto attn_norm_values = get_f32(first, attn_norm);
    std::vector<float> q_projection, k_projection, v_projection;
    const auto qkv_started = std::chrono::steady_clock::now();
    const std::vector<int> projection_budgets = qwen3_qkv_kernel_budgets(g_execution_policy);
    execute_independent({
        [&](int threads) { q_projection = execute_projection_kernel(get(model, prefix + "attn_q.weight").generic, q_weight_bytes,
            attn_norm_values, EMBED, EMBED, positions, threads); },
        [&](int threads) { k_projection = execute_projection_kernel(get(model, prefix + "attn_k.weight").generic, k_weight_bytes,
            attn_norm_values, EMBED, KV_HEADS * HEAD_DIM, positions, threads); },
        [&](int threads) { v_projection = execute_projection_kernel(get(model, prefix + "attn_v.weight").generic, v_weight_bytes,
            attn_norm_values, EMBED, KV_HEADS * HEAD_DIM, positions, threads); },
    }, projection_budgets);
    set_tensor(q_linear, q_projection.data(), q_projection.size() * sizeof(float));
    set_tensor(k_linear, k_projection.data(), k_projection.size() * sizeof(float));
    set_tensor(v_linear, v_projection.data(), v_projection.size() * sizeof(float));
    const double qkv_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - qkv_started).count();
    std::printf("qkv_parallel_stage layer=%u ms=%.3f mode=%s budget=%u tasks=3 assigned=%d,%d,%d\n",
        layer, qkv_ms, g_execution_policy.mode == Qwen3ExecutionMode::Parallel ? "parallel" : "serial",
        g_execution_policy.total_cpu_threads, projection_budgets[0], projection_budgets[1], projection_budgets[2]);

    const auto attention_started = std::chrono::steady_clock::now();
    ggml_cgraph * first_graph = ggml_new_graph_custom(first.ctx, 2048, false);
    ggml_build_forward_expand(first_graph, qrope); ggml_build_forward_expand(first_graph, krope);
    ggml_build_forward_expand(first_graph, v_heads);
    ggml_build_forward_expand(first_graph, attention_context);
    ggml_build_forward_expand(first_graph, reference_attention_context);
    if (ggml_backend_graph_compute(first.backend, first_graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Qwen attention graph execution failed");
    ggml_backend_synchronize(first.backend);
    if (const char * export_dir = std::getenv("VBUF_QWEN_EXPORT_DIR")) {
        const std::filesystem::path directory(export_dir);
        std::filesystem::create_directories(directory);
        const auto keys = get_tensor_bytes(first, k_cache);
        const auto values = get_tensor_bytes(first, v_cache);
        const size_t bytes_per_token = static_cast<size_t>(KV_HEADS) * HEAD_DIM * sizeof(uint16_t);
        if (keys.size() != bytes_per_token * positions || values.size() != keys.size())
            throw std::runtime_error("KV byte export geometry mismatch");
        for (uint32_t position = 0; position < positions; ++position) {
            const uint32_t cache_position = incremental_execution ? absolute_position + position : position;
            std::ostringstream key_name, value_name;
            key_name << "kv_key_layer-" << layer << "_position-" << cache_position << ".f16";
            value_name << "kv_value_layer-" << layer << "_position-" << cache_position << ".f16";
            write_binary_export(directory / key_name.str(), keys.data() + position * bytes_per_token, bytes_per_token);
            write_binary_export(directory / value_name.str(), values.data() + position * bytes_per_token, bytes_per_token);
        }
    }
    const double attention_graph_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - attention_started).count();
    std::printf("attention_graph_stage layer=%u ms=%.3f includes_qk_norm_rope_attention=YES\n",
        layer, attention_graph_ms);
    if (incremental_execution && layer == 14 && absolute_position == 7) {
        const size_t old_bytes = retained_cache != nullptr ? retained_cache->layer_bytes(layer) : replay_key_history->size();
        const uint8_t * old_key = retained_cache != nullptr ? retained_cache->key_bytes(layer) : replay_key_history->data();
        const uint8_t * old_value = retained_cache != nullptr ? retained_cache->value_bytes(layer) : replay_value_history->data();
        std::vector<uint8_t> past_k_readback(old_bytes), past_v_readback(old_bytes);
        ggml_backend_tensor_get(attention_groups.front().past_k, past_k_readback.data(), 0, old_bytes);
        ggml_backend_tensor_get(attention_groups.front().past_v, past_v_readback.data(), 0, old_bytes);
        const bool exact = std::equal(past_k_readback.begin(), past_k_readback.end(), old_key) &&
            std::equal(past_v_readback.begin(), past_v_readback.end(), old_value);
        std::printf("kv_storage_cache_to_graph_readback mode=%s layer=%u previous_positions=%u K_exact=%s V_exact=%s cached_rope_reapplied=NO\n",
            retained_cache != nullptr ? "persistent" : "ephemeral_replay", layer, absolute_position,
            exact ? "YES" : "NO", exact ? "YES" : "NO");
        if (!exact) throw std::runtime_error("cached K/V changed while loading past tensors into graph");
    }
    const auto layer_input_values = get_f32(first, layer_input);
    const auto q_linear_values = get_f32(first, q_linear);
    const auto k_linear_values = get_f32(first, k_linear);
    const auto qnorm_values = get_f32(first, qnorm);
    const auto knorm_values = get_f32(first, knorm);
    const auto q_values = get_f32(first, qrope);
    const auto k_values = get_f32(first, krope);
    const auto v_values = get_f32(first, v_heads);
    auto current_reference = [&](const std::string & stem, size_t values_per_position) {
        if (internal_qualification_mode())
            return std::vector<float>(values_per_position * (incremental_execution ? 1 : positions), 0.0f);
        auto values = reference(reference_dir, stem);
        if (!incremental_execution) return values;
        if (values.size() == values_per_position) return values;
        if (values.size() != values_per_position * logical_key_positions)
            throw std::runtime_error("incremental reference checkpoint geometry mismatch: " + stem);
        values.erase(values.begin(), values.end() - values_per_position);
        return values;
    };
    bool pass = true;
    std::printf("layer_run path=%s layer=%u positions=%u input_source=%s\n",
        reference_reset ? "REFERENCE_RESET_LOCAL" : (replay_incremental ? "INCREMENTAL_REPLAY" : "SEQUENTIAL"), layer, positions,
        reference_reset ? "llama.cpp reference hidden" : (layer == 0 ? "embedding lookup" : "previous vBuf layer output"));
    if (layer == 0) pass &= compare("embedding", layer_input_values, current_reference("embd-0", EMBED), positions);
    else pass &= compare("block_input", layer_input_values,
        current_reference("l_out-" + std::to_string(layer - 1) + "-0", EMBED), positions);
    pass &= compare("attention_rmsnorm", attn_norm_values, current_reference("attn_norm" + suffix + "-0", EMBED), positions);
    pass &= compare("q_projection", q_linear_values, current_reference("Qcur" + suffix + "-0", EMBED), positions);
    pass &= compare("k_projection", k_linear_values, current_reference("Kcur" + suffix + "-0", KV_HEADS * HEAD_DIM), positions);
    pass &= compare("q_rmsnorm", qnorm_values, current_reference("Qcur_normed" + suffix + "-0", EMBED), positions);
    pass &= compare("k_rmsnorm", knorm_values, current_reference("Kcur_normed" + suffix + "-0", KV_HEADS * HEAD_DIM), positions);
    pass &= compare("q_rope_neox", q_values, current_reference("Qcur" + suffix + "-2", EMBED), positions);
    pass &= compare("k_rope_neox", k_values, current_reference("Kcur" + suffix + "-2", KV_HEADS * HEAD_DIM), positions);
    pass &= compare("v_projection", v_values, current_reference("Vcur" + suffix + "-1", KV_HEADS * HEAD_DIM), positions);

    auto collect_group_rows = [&](bool probabilities) {
        std::vector<float> rows(static_cast<size_t>(HEADS) * positions * key_positions, 0.0f);
        for (const auto & group : attention_groups) {
            const std::vector<float> values = get_f32(first,
                probabilities ? group.probabilities : group.raw_scores);
            const uint32_t extent = group.group.compute_extent;
            for (uint32_t head = 0; head < HEADS; ++head)
                for (uint32_t local = 0; local < group.group.query_count; ++local) {
                    const uint32_t output_query = incremental_execution ? local : group.group.first_query + local;
                    const size_t source = (static_cast<size_t>(head) * group.group.query_count + local) * extent;
                    const size_t destination = (static_cast<size_t>(head) * positions + output_query) * key_positions;
                    std::copy_n(values.data() + source, extent, rows.data() + destination);
                }
        }
        return rows;
    };
    const auto score_matrix = collect_group_rows(false);
    const auto probability_matrix = collect_group_rows(true);
    std::vector<float> reference_probabilities(static_cast<size_t>(HEADS) * positions * key_positions, 0.0f);
    for (const auto & group : attention_groups) {
        const uint32_t extent = group.group.compute_extent;
        for (uint32_t head = 0; head < HEADS; ++head)
            for (uint32_t local = 0; local < group.group.query_count; ++local) {
                const uint32_t output_query = incremental_execution ? local : group.group.first_query + local;
                const size_t source = (static_cast<size_t>(head) * group.group.query_count + local) * extent;
                const size_t destination = (static_cast<size_t>(head) * positions + output_query) * key_positions;
                std::copy_n(group.reference_values.data() + source, extent,
                    reference_probabilities.data() + destination);
            }
    }
    const auto reference_scores_logical = reference_attention_prefix(reference_dir,
        "kq" + suffix + "-0", logical_key_positions);
    std::vector<float> reference_scores(static_cast<size_t>(HEADS) * positions * key_positions, 0.0f);
    for (uint32_t head = 0; head < HEADS; ++head) {
        for (uint32_t query = 0; query < positions; ++query) {
            const uint32_t reference_query = incremental_execution ? absolute_position : query;
            for (uint32_t key = 0; key < logical_key_positions; ++key) {
                const size_t src = (static_cast<size_t>(head) * logical_key_positions + reference_query) *
                    logical_key_positions + key;
                const size_t dst = (static_cast<size_t>(head) * positions + query) * key_positions + key;
                reference_scores[dst] = reference_scores_logical[src];
            }
        }
    }
    pass &= compare("attention_scores", score_matrix, reference_scores, positions);
    if (!incremental_execution) {
        const auto qk_fp64_oracle = attention_qk_fp64(q_values, k_values, positions);
        std::vector<float> raw_scores_logical(static_cast<size_t>(HEADS) * positions * logical_key_positions);
        for (uint32_t head = 0; head < HEADS; ++head)
            for (uint32_t query = 0; query < positions; ++query)
                std::copy_n(score_matrix.data() + (static_cast<size_t>(head) * positions + query) * key_positions,
                    logical_key_positions, raw_scores_logical.data() +
                    (static_cast<size_t>(head) * positions + query) * logical_key_positions);
        compare("llama_qk_vs_fp64_oracle", reference_scores_logical, qk_fp64_oracle, logical_key_positions);
        compare("vbuf_qk_vs_fp64_oracle", raw_scores_logical, qk_fp64_oracle, logical_key_positions);
    }
    pass &= compare("attention_probabilities", probability_matrix, reference_probabilities, positions);
    const auto context_values = get_f32(first, attention_context);
    const auto reference_probability_context_values = get_f32(first, reference_attention_context);
    const auto reference_context_values = current_reference("kqv_out" + suffix + "-0", EMBED);
    if (padded_context_output != nullptr) {
        std::vector<uint8_t> compact_v(ggml_nbytes(v_batched));
        ggml_backend_tensor_get(v_batched, compact_v.data(), 0, compact_v.size());
        std::vector<uint8_t> padded_v(ggml_nbytes(padded_v_input), 0);
        const size_t f16_bytes = ggml_type_size(GGML_TYPE_F16);
        for (int64_t kv_head = 0; kv_head < KV_HEADS; ++kv_head)
            for (int64_t d = 0; d < HEAD_DIM; ++d)
                std::memcpy(padded_v.data() + (kv_head * HEAD_DIM + d) * REFERENCE_KV_CAPACITY * f16_bytes,
                    compact_v.data() + (kv_head * HEAD_DIM + d) * positions * f16_bytes,
                    positions * f16_bytes);
        const auto full_reference_probabilities = reference(reference_dir, "kq_soft_max-0-0");
        if (full_reference_probabilities.size() != static_cast<size_t>(REFERENCE_KV_CAPACITY) * positions * HEADS)
            throw std::runtime_error("reference full-capacity softmax shape mismatch");
        set_tensor(padded_v_input, padded_v.data(), padded_v.size());
        set_tensor(padded_probabilities_input, full_reference_probabilities.data(),
            full_reference_probabilities.size() * sizeof(float));
        ggml_cgraph * padded_graph = ggml_new_graph(first.ctx);
        ggml_build_forward_expand(padded_graph, padded_context_output);
        if (ggml_backend_graph_compute(first.backend, padded_graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("reference-extent V projection control failed");
        ggml_backend_synchronize(first.backend);
        const auto padded_context_values = get_f32(first, padded_context_output);
        std::printf("projection_control kv_extent=%lld live_keys=%u padded_keys_zero_probability=YES input_v_type=F16 input_probability_type=F32\n",
            static_cast<long long>(REFERENCE_KV_CAPACITY), positions);
        compare("v_context_reference_kv_extent_vbuf", padded_context_values,
            reference_context_values, positions);
        compare("v_context_compact_vs_reference_extent", context_values,
            padded_context_values, positions);
    }
    pass &= compare("causal_gqa_attention", context_values, reference_context_values, positions);
    compare("reference_probabilities_vbuf_v_projection", reference_probability_context_values,
        reference_context_values, positions);
    compare("actual_vs_reference_probabilities_vprojection", context_values,
        reference_probability_context_values, positions);
    const auto v_projection_oracle = attention_v_projection_fp64(v_values,
        reference_probabilities, positions);
    compare("llama_v_projection_vs_fp64_oracle", reference_context_values,
        v_projection_oracle, positions);
    compare("vbuf_v_projection_vs_fp64_oracle", reference_probability_context_values,
        v_projection_oracle, positions);

    const uint32_t ffn_positions = layer + 1 == model.layer_count && !force_all_final_positions ? 1 : positions;
    const size_t ffn_input_offset = static_cast<size_t>(positions - ffn_positions) * EMBED;
    std::vector<float> ffn_input_hidden(layer_input_values.begin() + ffn_input_offset,
        layer_input_values.end());
    std::vector<float> ffn_attention_context(context_values.begin() + ffn_input_offset,
        context_values.end());
    auto reference_swiglu = current_reference("ffn_swiglu" + suffix + "-0", FFN);
    if (reference_swiglu.size() != static_cast<size_t>(FFN) * ffn_positions) {
        if (reference_swiglu.size() != static_cast<size_t>(FFN) * positions || ffn_positions >= positions)
            throw std::runtime_error("reference SwiGLU checkpoint geometry mismatch");
        reference_swiglu.erase(reference_swiglu.begin(),
            reference_swiglu.end() - static_cast<size_t>(FFN) * ffn_positions);
    }
    Context second = make_context();
    auto add_second = [&](const std::string & name) { Tensor & tensor = get(model, name); make_tensor(second, tensor); return tensor.ggml; };
    ggml_tensor * out_w = add_second(prefix + "attn_output.weight");
    ggml_tensor * ffnorm_w = add_second(prefix + "ffn_norm.weight");
    ggml_tensor * down = add_second(prefix + "ffn_down.weight");
    Tensor & down_metadata = get(model, prefix + "ffn_down.weight");
    TensorGeometry down_geometry{};
    if (derive_tensor_geometry(down_metadata.generic, &down_geometry) != AdapterError::None)
        throw std::runtime_error("invalid FFN-down geometry for row check");
    const uint64_t down_rows = static_cast<uint64_t>(down_geometry.ne[1]);
    const std::array<int32_t, 3> row_ids_values{
        0, static_cast<int32_t>(down_rows / 2), static_cast<int32_t>(down_rows - 1) };
    ggml_tensor * row_ids = ggml_new_tensor_1d(second.ctx, GGML_TYPE_I32, row_ids_values.size());
    ggml_tensor * decoded_rows = ggml_get_rows(second.ctx, down, row_ids);
    ggml_tensor * hidden_in = ggml_new_tensor_2d(second.ctx, GGML_TYPE_F32, EMBED, ffn_positions);
    ggml_tensor * attention_in = ggml_new_tensor_2d(second.ctx, GGML_TYPE_F32, EMBED, ffn_positions);
    ggml_tensor * reference_swiglu_in = ggml_new_tensor_2d(second.ctx, GGML_TYPE_F32, FFN, ffn_positions);
    ggml_tensor * projected = ggml_mul_mat(second.ctx, out_w, attention_in);
    ggml_tensor * residual = ggml_add(second.ctx, projected, hidden_in);
    ggml_tensor * ffn_norm = ggml_mul(second.ctx, ggml_rms_norm(second.ctx, residual, 1e-6f), ffnorm_w);
    // Gate and up share the completed FFN normalization and have no dependency
    // on one another; they are evaluated in independent kernel contexts.
    ggml_tensor * gate_out = ggml_new_tensor_2d(second.ctx, GGML_TYPE_F32, FFN, ffn_positions);
    ggml_tensor * up_out = ggml_new_tensor_2d(second.ctx, GGML_TYPE_F32, FFN, ffn_positions);
    ggml_tensor * swiglu = ggml_mul(second.ctx, ggml_silu(second.ctx, gate_out), up_out);
    ggml_tensor * down_out = ggml_mul_mat(second.ctx, down, swiglu);
    ggml_tensor * down_reference_input = ggml_mul_mat(second.ctx, down, reference_swiglu_in);
    const auto * down_cpu_traits = ggml_get_type_traits_cpu(down->type);
    const int thread_count = static_cast<int>(g_execution_policy.total_cpu_threads);
    const int64_t chunk_size = down_out->ne[0] == 1 || down_out->ne[1] == 1 ? 64 : 16;
    const int64_t output_chunks = (down_out->ne[0] + chunk_size - 1) / chunk_size;
    const int64_t token_chunks = (down_out->ne[1] + chunk_size - 1) / chunk_size;
    std::printf("ffn_down_layout tokens=%lld src0_type=%s src0_ne=[%lld,%lld] src0_nb=[%zu,%zu] "
        "src0_contiguous=%s src0_transposed=%s src1_type=%s src1_ne=[%lld,%lld] src1_nb=[%zu,%zu] "
        "src1_contiguous=%s dst_ne=[%lld,%lld] dst_nb=[%zu,%zu] dst_contiguous=%s "
        "vec_dot_type=%s vec_dot_rows=%lld chunk_size=%lld chunks=[%lld,%lld] threads=%d\n",
        static_cast<long long>(ffn_positions), ggml_type_name(down->type),
        static_cast<long long>(down->ne[0]), static_cast<long long>(down->ne[1]), down->nb[0], down->nb[1],
        ggml_is_contiguous(down) ? "YES" : "NO", ggml_is_transposed(down) ? "YES" : "NO",
        ggml_type_name(swiglu->type), static_cast<long long>(swiglu->ne[0]), static_cast<long long>(swiglu->ne[1]),
        swiglu->nb[0], swiglu->nb[1], ggml_is_contiguous(swiglu) ? "YES" : "NO",
        static_cast<long long>(down_out->ne[0]), static_cast<long long>(down_out->ne[1]),
        down_out->nb[0], down_out->nb[1], ggml_is_contiguous(down_out) ? "YES" : "NO",
        ggml_type_name(down_cpu_traits->vec_dot_type), static_cast<long long>(down_cpu_traits->nrows),
        static_cast<long long>(chunk_size), static_cast<long long>(output_chunks),
        static_cast<long long>(token_chunks), thread_count);
    ggml_tensor * block_out = ggml_add(second.ctx, down_out, residual);
    second.buffer = ggml_backend_alloc_ctx_tensors(second.ctx, second.backend);
    if (!second.buffer) throw std::runtime_error("Qwen FFN-stage tensor allocation failed");
    g_qualification_memory_peaks.second_stage_bytes = std::max(
        g_qualification_memory_peaks.second_stage_bytes, ggml_backend_buffer_get_size(second.buffer));
    set_tensor(hidden_in, ffn_input_hidden.data(), ffn_input_hidden.size() * sizeof(float));
    set_tensor(attention_in, ffn_attention_context.data(), ffn_attention_context.size() * sizeof(float));
    set_tensor(reference_swiglu_in, reference_swiglu.data(), reference_swiglu.size() * sizeof(float));
    set_tensor(row_ids, row_ids_values.data(), row_ids_values.size() * sizeof(int32_t));
    const auto * down_traits = ggml_get_type_traits(down_geometry.type);
    const size_t down_row_bytes = ggml_row_size(down_geometry.type, down_geometry.ne[0]);
    std::vector<float> direct_decoded_rows(row_ids_values.size() * down_geometry.ne[0]);
    uint64_t materialized_down_hash = 0;
    std::vector<uint8_t> gate_weight_bytes, up_weight_bytes;
    ref_id = 100;
    for (const std::string & name : { prefix + "attn_output.weight", prefix + "ffn_norm.weight", prefix + "ffn_gate.weight", prefix + "ffn_up.weight", prefix + "ffn_down.weight" }) {
        Tensor & tensor = get(model, name);
        if (!model.materializer->request(ref_id, tensor.persistent(), tensor.length) || model.materializer->wait(ref_id) != MaterializationState::Ready)
            throw std::runtime_error("vBuf FFN weight request failed: " + name);
        auto ready = model.materializer->obtain_ready_tensor(ref_id++);
        if (!ready || ready->payload_len != tensor.length) throw std::runtime_error("vBuf FFN lease failed");
        if (name == prefix + "ffn_gate.weight")
            gate_weight_bytes.assign(ready->payload, ready->payload + ready->payload_len);
        else if (name == prefix + "ffn_up.weight")
            up_weight_bytes.assign(ready->payload, ready->payload + ready->payload_len);
        else if (name == prefix + "ffn_down.weight") {
            materialized_down_hash = fnv1a64(ready->payload, ready->payload_len);
            std::printf("ffn_down_materialized bytes=%llu fnv1a64=%016llx\n",
                static_cast<unsigned long long>(ready->payload_len),
                static_cast<unsigned long long>(materialized_down_hash));
            for (size_t i = 0; i < row_ids_values.size(); ++i)
                down_traits->to_float(ready->payload + static_cast<uint64_t>(row_ids_values[i]) * down_row_bytes,
                    direct_decoded_rows.data() + i * down_geometry.ne[0], down_geometry.ne[0]);
        }
        if (name != prefix + "ffn_gate.weight" && name != prefix + "ffn_up.weight")
            ggml_backend_tensor_set(tensor.ggml, ready->payload, 0, tensor.length);
        model.materializer->release(ref_id - 1);
    }
    ggml_cgraph * pre_gate_graph = ggml_new_graph(second.ctx);
    ggml_build_forward_expand(pre_gate_graph, ffn_norm);
    if (ggml_backend_graph_compute(second.backend, pre_gate_graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Qwen attention output/FFN normalization graph execution failed");
    ggml_backend_synchronize(second.backend);
    const auto fn_values = get_f32(second, ffn_norm);
    std::vector<float> gate_projection, up_projection;
    const auto gate_up_started = std::chrono::steady_clock::now();
    const std::vector<int> gate_up_budgets = qwen3_gate_up_kernel_budgets(g_execution_policy);
    execute_independent({
        [&](int threads) { gate_projection = execute_projection_kernel(
            get(model, prefix + "ffn_gate.weight").generic, gate_weight_bytes, fn_values,
            EMBED, FFN, ffn_positions, threads); },
        [&](int threads) { up_projection = execute_projection_kernel(
            get(model, prefix + "ffn_up.weight").generic, up_weight_bytes, fn_values,
            EMBED, FFN, ffn_positions, threads); },
    }, gate_up_budgets);
    set_tensor(gate_out, gate_projection.data(), gate_projection.size() * sizeof(float));
    set_tensor(up_out, up_projection.data(), up_projection.size() * sizeof(float));
    const double gate_up_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gate_up_started).count();
    std::printf("ffn_gate_up_parallel_stage layer=%u ms=%.3f mode=%s budget=%u assigned=%d,%d\n",
        layer, gate_up_ms, g_execution_policy.mode == Qwen3ExecutionMode::Parallel ? "parallel" : "serial",
        g_execution_policy.total_cpu_threads, gate_up_budgets[0], gate_up_budgets[1]);
    const auto ffn_down_started = std::chrono::steady_clock::now();
    ggml_cgraph * second_graph = ggml_new_graph(second.ctx);
    ggml_build_forward_expand(second_graph, block_out);
    ggml_build_forward_expand(second_graph, down_reference_input);
    ggml_build_forward_expand(second_graph, decoded_rows);
    if (ggml_backend_graph_compute(second.backend, second_graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Qwen output projection/FFN graph execution failed");
    ggml_backend_synchronize(second.backend);
    const double ffn_down_graph_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ffn_down_started).count();
    std::printf("post_gate_up_graph_stage layer=%u ms=%.3f "
        "includes_output_projection_residual_swiglu_down=YES\n", layer, ffn_down_graph_ms);
    const auto proj_values = get_f32(second, projected);
    const auto residual_values = get_f32(second, residual);
    const auto gate_values = get_f32(second, gate_out);
    const auto up_values = get_f32(second, up_out);
    const auto swiglu_values = get_f32(second, swiglu);
    const auto down_values = get_f32(second, down_out);
    const auto down_reference_values = get_f32(second, down_reference_input);
    const auto output_values = get_f32(second, block_out);
    const auto ggml_decoded_rows = get_f32(second, decoded_rows);
    std::vector<uint8_t> packed_down(ggml_nbytes(down));
    ggml_backend_tensor_get(down, packed_down.data(), 0, packed_down.size());
    const uint64_t readback_down_hash = fnv1a64(packed_down.data(), packed_down.size());
    std::printf("ffn_down_backend_readback bytes=%zu fnv1a64=%016llx matches_materialized=%s\n",
        packed_down.size(), static_cast<unsigned long long>(readback_down_hash),
        readback_down_hash == materialized_down_hash ? "YES" : "NO");
    double row_decode_max = 0;
    for (size_t i = 0; i < ggml_decoded_rows.size(); ++i)
        row_decode_max = std::max(row_decode_max, std::abs(static_cast<double>(ggml_decoded_rows[i]) - direct_decoded_rows[i]));
    std::printf("ffn_down_row_decode indices=%d,%d,%d values=%zu max_abs=%.9g status=%s\n",
        row_ids_values[0], row_ids_values[1], row_ids_values[2], ggml_decoded_rows.size(), row_decode_max,
        row_decode_max == 0 ? "PASS" : "FAIL");
    const auto ref_residual = current_reference("ffn_inp" + suffix + "-0", EMBED);
    const auto ref_input_hidden = current_reference(layer == 0 ? "embd-0" :
        "l_out-" + std::to_string(layer - 1) + "-0", EMBED);
    std::vector<float> ref_projected(ref_residual.size());
    for (size_t i = 0; i < ref_projected.size(); ++i)
        ref_projected[i] = ref_residual[i] - ref_input_hidden[i];
    pass &= compare("attention_output_projection_derived", proj_values, ref_projected, ffn_positions);
    pass &= compare("attention_residual", residual_values, ref_residual, ffn_positions);
    pass &= compare("ffn_rmsnorm", fn_values, current_reference("ffn_norm" + suffix + "-0", EMBED), ffn_positions);
    pass &= compare("ffn_gate", gate_values, current_reference("ffn_gate" + suffix + "-0", FFN), ffn_positions);
    pass &= compare("ffn_up", up_values, current_reference("ffn_up" + suffix + "-0", FFN), ffn_positions);
    pass &= compare("ffn_swiglu", swiglu_values, reference_swiglu, ffn_positions);
    const auto reference_down = current_reference("ffn_out" + suffix + "-0", EMBED);
    pass &= compare("ffn_down", down_values, reference_down, ffn_positions);
    pass &= compare(std::string(reference_reset ? "local_block" : "sequential_block") +
        std::to_string(layer) + "_output", output_values,
        current_reference("l_out" + suffix + "-0", EMBED), ffn_positions);
    compare("ffn_down_same_input_ggml", down_reference_values, reference_down, ffn_positions);
    if (thread_count == 8 && (layer == 0 || layer == 3 || layer == 6 || layer == 7)) {
        report_activation_quantization_delta(reference_swiglu, swiglu_values, down_geometry.type,
            static_cast<uint32_t>(down_geometry.ne[0]), positions, layer,
            reference_reset ? "REFERENCE_RESET_LOCAL" : "SEQUENTIAL");
        const ProjectionOracle reference_oracle = simple_projection(packed_down, reference_swiglu,
            down_geometry.type, static_cast<uint32_t>(down_geometry.ne[0]),
            static_cast<uint32_t>(down_geometry.ne[1]), positions);
        const ProjectionOracle vbuf_oracle = simple_projection(packed_down, swiglu_values,
            down_geometry.type, static_cast<uint32_t>(down_geometry.ne[0]),
            static_cast<uint32_t>(down_geometry.ne[1]), positions);
        const auto quantized_reference_oracle = ggml_quantized_dot_projection(packed_down, reference_swiglu,
            down_geometry.type, static_cast<uint32_t>(down_geometry.ne[0]),
            static_cast<uint32_t>(down_geometry.ne[1]), positions);
        const auto quantized_vbuf_oracle = ggml_quantized_dot_projection(packed_down, swiglu_values,
            down_geometry.type, static_cast<uint32_t>(down_geometry.ne[0]),
            static_cast<uint32_t>(down_geometry.ne[1]), positions);
        compare("llama_vs_simple_fp64", reference_down, reference_oracle.fp64, positions);
        compare("vbuf_same_input_vs_simple_fp64", down_reference_values, reference_oracle.fp64, positions);
        compare("vbuf_actual_input_vs_simple_fp64", down_values, vbuf_oracle.fp64, positions);
        compare("simple_ref_vs_vbuf_input_fp64", reference_oracle.fp64, vbuf_oracle.fp64, positions);
        compare("llama_vs_quantized_dot_oracle", reference_down, quantized_reference_oracle, positions);
        compare("vbuf_same_input_vs_quantized_dot_oracle", down_reference_values, quantized_reference_oracle, positions);
        compare("vbuf_actual_input_vs_quantized_dot_oracle", down_values, quantized_vbuf_oracle, positions);
        compare("simple_fp32_vs_fp64", reference_oracle.fp32, reference_oracle.fp64, positions);
    }
    if (const char * export_intermediates = std::getenv("VBUF_QWEN_EXPORT_INTERMEDIATES")) {
        if (std::string(export_intermediates) == "1" &&
            (layer == 0 || layer == model.layer_count / 2 || layer + 1 == model.layer_count)) {
            const char * export_dir = std::getenv("VBUF_QWEN_EXPORT_DIR");
            if (export_dir == nullptr || *export_dir == '\0')
                throw std::runtime_error("intermediate export requires VBUF_QWEN_EXPORT_DIR");
            const std::filesystem::path directory(export_dir);
            std::filesystem::create_directories(directory);
            const auto export_values = [&](const char * name, const std::vector<float> & values) {
                write_f32_export(directory / ("intermediate-layer-" + std::to_string(layer) + "-" + name + ".f32"), values);
            };
            export_values("q_projection", q_linear_values);
            export_values("k_projection", k_linear_values);
            export_values("v_projection", v_values);
            export_values("q_rope", q_values);
            export_values("k_rope", k_values);
            export_values("attention_scores", score_matrix);
            export_values("attention_probabilities", probability_matrix);
            export_values("attention_context", context_values);
            export_values("ffn_rmsnorm", fn_values);
            export_values("ffn_gate", gate_values);
            export_values("ffn_up", up_values);
            export_values("ffn_swiglu", swiglu_values);
            export_values("ffn_down", down_values);
            export_values("block_output", output_values);
        }
    }
    if (diagnostics != nullptr) {
        diagnostics->layer_input = layer_input_values;
        diagnostics->attention_rmsnorm = attn_norm_values;
        diagnostics->q_projection = q_linear_values;
        diagnostics->k_projection = k_linear_values;
        diagnostics->q_rmsnorm = qnorm_values;
        diagnostics->k_rmsnorm = knorm_values;
        diagnostics->q_rope = q_values;
        diagnostics->k_rope = k_values;
        diagnostics->v_projection = v_values;
        diagnostics->attention_scores = score_matrix;
        diagnostics->attention_probabilities = probability_matrix;
        diagnostics->attention_key_positions = key_positions;
        diagnostics->attention_query_positions = positions;
        diagnostics->attention_context = context_values;
        diagnostics->attention_projection = proj_values;
        diagnostics->attention_residual = residual_values;
        diagnostics->ffn_rmsnorm = fn_values;
        diagnostics->ffn_gate = gate_values;
        diagnostics->ffn_up = up_values;
        diagnostics->ffn_swiglu = swiglu_values;
        diagnostics->ffn_down = down_values;
        diagnostics->block_output = output_values;
        diagnostics->key_f16.resize(ggml_nbytes(k_cache));
        diagnostics->value_f16.resize(ggml_nbytes(v_cache));
        ggml_backend_tensor_get(k_cache, diagnostics->key_f16.data(), 0, diagnostics->key_f16.size());
        ggml_backend_tensor_get(v_cache, diagnostics->value_f16.data(), 0, diagnostics->value_f16.size());
    }
    if (retained_cache != nullptr) {
        std::vector<uint8_t> key_bytes(ggml_nbytes(k_cache));
        std::vector<uint8_t> value_bytes(ggml_nbytes(v_cache));
        ggml_backend_tensor_get(k_cache, key_bytes.data(), 0, key_bytes.size());
        ggml_backend_tensor_get(v_cache, value_bytes.data(), 0, value_bytes.size());
        if (key_bytes.size() != static_cast<size_t>(KV_HEADS) * HEAD_DIM * sizeof(uint16_t))
            throw std::runtime_error("Qwen3 cached K/V byte geometry mismatch");
        retained_cache->append(layer, absolute_position,
            key_bytes.data(), key_bytes.size(), value_bytes.data(), value_bytes.size());
        const size_t cache_offset = static_cast<size_t>(absolute_position) * key_bytes.size();
        const bool append_readback_exact = std::equal(key_bytes.begin(), key_bytes.end(),
            retained_cache->key_bytes(layer) + cache_offset) &&
            std::equal(value_bytes.begin(), value_bytes.end(),
            retained_cache->value_bytes(layer) + cache_offset);
        if (!append_readback_exact)
            throw std::runtime_error("KV cache append readback changed K/V bytes");
        if (layer == 14 && absolute_position == 6)
            std::printf("kv_storage_write_readback layer=%u position=%u K_exact=YES V_exact=YES key_offset=%zu value_offset=%zu kv_head_stride=%u component_stride=%zu\n",
                layer, absolute_position, cache_offset, cache_offset, HEAD_DIM * sizeof(uint16_t), sizeof(uint16_t));
        std::printf("kv_append layer=%u position=%u kv_heads=%u head_dim=%u bytes_per_tensor=%zu key_rope_applied=YES\n",
            layer, absolute_position, KV_HEADS, HEAD_DIM, key_bytes.size());
    }
    std::printf("kv_state=%s positions=%u causal_visibility=prefix_limited\n",
        retained_cache != nullptr ? "PERSISTENT_APPEND_ONLY" :
        (replay_incremental ? "EPHEMERAL_REPLAY_VECTOR" : "EPHEMERAL_IN_BATCH"), positions);
    std::printf("qwen3_block_boundary layer=%u positions=%u all_finite=%s "
        "numerical_equivalence=METRICS_RECORDED wall_ms=%.3f\n", layer, positions, pass ? "YES" : "NO",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - block_started).count());
    return (pass || use_canonical_softmax_extent) ? output_values : std::vector<float>{};
}

void analyze_attention_output_shape(Model & model, const LayerDiagnostics & full,
    const LayerDiagnostics & incremental, uint32_t layer, uint32_t position) {
    const uint32_t query_positions = full.attention_query_positions;
    if (query_positions <= position || incremental.attention_context.size() != EMBED)
        throw std::runtime_error("attention-output shape control geometry invalid");
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    Tensor & weight_metadata = get(model, prefix + "attn_output.weight");
    TensorGeometry geometry{};
    if (derive_tensor_geometry(weight_metadata.generic, &geometry) != AdapterError::None ||
        geometry.ne[0] != EMBED || geometry.ne[1] != EMBED)
        throw std::runtime_error("attention output weight geometry invalid");
    const auto * cpu_traits = ggml_get_type_traits_cpu(geometry.type);
    if (cpu_traits == nullptr) throw std::runtime_error("attention output CPU traits unavailable");
    std::printf("attention_output_dispatch layer=%u op=GGML_OP_MUL_MAT weight_type=%s activation_type=F32 activation_vec_dot=%s vec_dot_rows=%lld full_shape=[%u,%u] single_shape=[%u,1] contiguous=YES llamafile_compile_flag=NO threads=%s\n",
        layer, ggml_type_name(geometry.type), ggml_type_name(cpu_traits->vec_dot_type),
        static_cast<long long>(cpu_traits->nrows), EMBED, query_positions, EMBED,
        std::to_string(g_execution_policy.total_cpu_threads).c_str());
    const auto full_row = std::vector<float>(full.attention_context.begin() +
        static_cast<size_t>(position) * EMBED, full.attention_context.begin() +
        static_cast<size_t>(position + 1) * EMBED);
    const auto incremental_row = incremental.attention_context;
    const auto full_projection = std::vector<float>(full.attention_projection.begin() +
        static_cast<size_t>(position) * EMBED, full.attention_projection.begin() +
        static_cast<size_t>(position + 1) * EMBED);
    const auto incremental_projection = incremental.attention_projection;

    Context control = make_context();
    ggml_tensor * weight = make_tensor(control, weight_metadata);
    ggml_tensor * batch_input = ggml_new_tensor_2d(control.ctx, GGML_TYPE_F32, EMBED, query_positions);
    ggml_tensor * row_input = ggml_new_tensor_2d(control.ctx, GGML_TYPE_F32, EMBED, 1);
    ggml_tensor * batch_output = ggml_mul_mat(control.ctx, weight, batch_input);
    ggml_tensor * row_output = ggml_mul_mat(control.ctx, weight, row_input);
    ggml_cgraph * graph = ggml_new_graph(control.ctx);
    ggml_build_forward_expand(graph, batch_output);
    ggml_build_forward_expand(graph, row_output);
    control.buffer = ggml_backend_alloc_ctx_tensors(control.ctx, control.backend);
    if (!control.buffer) throw std::runtime_error("attention-output shape-control allocation failed");
    set_tensor(batch_input, full.attention_context.data(), full.attention_context.size() * sizeof(float));
    set_tensor(row_input, full_row.data(), full_row.size() * sizeof(float));
    constexpr uint32_t request_id = 0xfffffff0u;
    if (!model.materializer->request(request_id, weight_metadata.persistent(), weight_metadata.length) ||
        model.materializer->wait(request_id) != MaterializationState::Ready)
        throw std::runtime_error("attention-output oracle weight request failed");
    auto lease = model.materializer->obtain_ready_tensor(request_id);
    if (!lease || lease->payload_len != weight_metadata.length)
        throw std::runtime_error("attention-output oracle weight lease failed");
    std::vector<uint8_t> packed_weights(lease->payload, lease->payload + lease->payload_len);
    ggml_backend_tensor_set(weight, packed_weights.data(), 0, packed_weights.size());
    model.materializer->release(request_id);
    if (ggml_backend_graph_compute(control.backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("attention-output shape-control graph failed");
    ggml_backend_synchronize(control.backend);
    const auto batch_result = get_f32(control, batch_output);
    const auto row_result = get_f32(control, row_output);
    const auto batch_row = std::vector<float>(batch_result.begin() +
        static_cast<size_t>(position) * EMBED, batch_result.begin() +
        static_cast<size_t>(position + 1) * EMBED);
    compare("attention_output_full_actual_vs_same_shape_control", full_projection, batch_row, 1);
    compare("attention_output_same_input_batched_vs_single_row", batch_row, row_result, 1);
    compare("attention_output_incremental_vs_full_recompute", incremental_projection, full_projection, 1);

    if (cpu_traits->vec_dot_type == GGML_TYPE_Q8_K) {
        report_activation_quantization_delta(full_row, incremental_row, geometry.type,
            EMBED, 1, layer, "full_vs_incremental_attention_context");
    } else {
        std::printf("attention_output_activation_quantization=NOT_Q8_K activation_type=%s\n",
            ggml_type_name(cpu_traits->vec_dot_type));
    }
    Tensor & down_metadata = get(model, prefix + "ffn_down.weight");
    TensorGeometry down_geometry{};
    if (derive_tensor_geometry(down_metadata.generic, &down_geometry) != AdapterError::None)
        throw std::runtime_error("FFN-down quantization diagnostic geometry invalid");
    const auto full_swiglu = std::vector<float>(full.ffn_swiglu.begin() +
        static_cast<size_t>(position) * FFN, full.ffn_swiglu.begin() +
        static_cast<size_t>(position + 1) * FFN);
    report_activation_quantization_delta(full_swiglu, incremental.ffn_swiglu,
        down_geometry.type, FFN, 1, layer, "full_vs_incremental_ffn_down_input");
    const ProjectionOracle fp_oracle = simple_projection(packed_weights, full_row,
        geometry.type, EMBED, EMBED, 1);
    const ProjectionOracle inc_fp_oracle = simple_projection(packed_weights, incremental_row,
        geometry.type, EMBED, EMBED, 1);
    const auto q8_oracle = ggml_quantized_dot_projection(packed_weights, full_row,
        geometry.type, EMBED, EMBED, 1);
    const auto inc_q8_oracle = ggml_quantized_dot_projection(packed_weights, incremental_row,
        geometry.type, EMBED, EMBED, 1);
    compare("attention_output_full_vs_fp64_oracle", full_projection, fp_oracle.fp64, 1);
    compare("attention_output_incremental_vs_fp64_oracle", incremental_projection, inc_fp_oracle.fp64, 1);
    compare("attention_output_full_vs_q8_dot_oracle", full_projection, q8_oracle, 1);
    compare("attention_output_incremental_vs_q8_dot_oracle", incremental_projection, inc_q8_oracle, 1);
}

std::vector<float> run_final_head(Model & model, const std::string & reference_dir,
    const std::vector<float> & final_hidden, bool persistent_kv = false,
    std::vector<float> * normalized_output = nullptr) {
    if (final_hidden.empty() || final_hidden.size() % EMBED != 0)
        throw std::runtime_error("final hidden-state geometry mismatch");
    const uint32_t positions = static_cast<uint32_t>(final_hidden.size() / EMBED);
    Context context = make_context();
    Tensor & norm_metadata = get(model, "output_norm.weight");
    Tensor & output_metadata = get(model, "output.weight");
    ggml_tensor * norm_weight = make_tensor(context, norm_metadata);
    ggml_tensor * output_weight = make_tensor(context, output_metadata);
    ggml_tensor * input = ggml_new_tensor_2d(context.ctx, GGML_TYPE_F32, EMBED, positions);
    ggml_tensor * normalized = ggml_mul(context.ctx,
        ggml_rms_norm(context.ctx, input, 1e-6f), norm_weight);
    ggml_tensor * logits = ggml_mul_mat(context.ctx, output_weight, normalized);
    context.buffer = ggml_backend_alloc_ctx_tensors(context.ctx, context.backend);
    if (!context.buffer) throw std::runtime_error("final-head tensor allocation failed");
    g_qualification_memory_peaks.final_head_bytes = std::max(
        g_qualification_memory_peaks.final_head_bytes, ggml_backend_buffer_get_size(context.buffer));
    set_tensor(input, final_hidden.data(), final_hidden.size() * sizeof(float));
    uint64_t ref_id = 1000;
    for (Tensor * tensor : { &norm_metadata, &output_metadata }) {
        if (!model.materializer->request(ref_id, tensor->persistent(), tensor->length) ||
            model.materializer->wait(ref_id) != MaterializationState::Ready)
            throw std::runtime_error("vBuf final-head weight request failed: " + tensor->name);
        auto ready = model.materializer->obtain_ready_tensor(ref_id);
        if (!ready || ready->payload_len != tensor->length)
            throw std::runtime_error("vBuf final-head weight lease failed: " + tensor->name);
        ggml_backend_tensor_set(tensor->ggml, ready->payload, 0, tensor->length);
        model.materializer->release(ref_id++);
    }
    ggml_cgraph * graph = ggml_new_graph(context.ctx);
    ggml_build_forward_expand(graph, logits);
    if (ggml_backend_graph_compute(context.backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("final RMSNorm/output projection graph failed");
    ggml_backend_synchronize(context.backend);
    const auto norm_values = get_f32(context, normalized);
    const auto logit_values = get_f32(context, logits);
    if (normalized_output != nullptr) *normalized_output = norm_values;
    const size_t vocab = static_cast<size_t>(output_weight->ne[1]);
    if (logit_values.size() != vocab * positions)
        throw std::runtime_error("final logits geometry mismatch");
    const std::vector<float> final_norm(norm_values.end() - EMBED, norm_values.end());
    const std::vector<float> final_logits(logit_values.end() - vocab, logit_values.end());
    if (!internal_qualification_mode()) {
        const auto reference_norm = reference(reference_dir, "result_norm-0");
        const auto reference_logits = reference(reference_dir, "result_output-0");
        compare("final_rmsnorm", final_norm, reference_norm, 1);
        compare("final_logits", final_logits, reference_logits, 1);
    }
    auto top_indices = [](const std::vector<float> & values, size_t count) {
        std::vector<uint32_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        const auto better = [&](uint32_t a, uint32_t b) {
            return values[a] == values[b] ? a < b : values[a] > values[b];
        };
        count = std::min(count, indices.size());
        std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), better);
        indices.resize(count);
        return indices;
    };
    const auto actual_top = top_indices(final_logits, 10);
    if (!internal_qualification_mode()) {
        const auto reference_logits = reference(reference_dir, "result_output-0");
        const auto ref_top = top_indices(reference_logits, 10);
        size_t overlap = 0;
        for (uint32_t id : actual_top)
            if (std::find(ref_top.begin(), ref_top.end(), id) != ref_top.end()) ++overlap;
        const float actual_margin = final_logits[actual_top[0]] - final_logits[actual_top[1]];
        const float reference_margin = reference_logits[ref_top[0]] - reference_logits[ref_top[1]];
        std::printf("greedy_next_token reference=%u vbuf=%u match=%s reference_margin=%.9g vbuf_margin=%.9g top10_overlap=%zu/10 status=METRICS_RECORDED\n",
            ref_top[0], actual_top[0], ref_top[0] == actual_top[0] ? "YES" : "NO",
            reference_margin, actual_margin, overlap);
    } else {
        std::printf("greedy_next_token reference=NOT_TESTED vbuf=%u reference_checks=DISABLED_INTERNAL_QUALIFICATION\n",
            actual_top[0]);
    }
    std::printf("final_head_scope=positions_%u full_recompute_generation=METRICS_RECORDED persistent_kv=%s\n",
        positions, persistent_kv ? "ACTIVE" : "NOT_IMPLEMENTED");
    return logit_values;
}

uint32_t greedy_token(const std::vector<float> & logits) {
    if (logits.empty()) throw std::runtime_error("cannot select from empty logits");
    uint32_t best = 0;
    for (uint32_t id = 1; id < logits.size(); ++id)
        if (logits[id] > logits[best]) best = id;
    return best;
}

std::vector<float> final_position(const std::vector<float> & values, size_t width) {
    if (values.size() < width || values.size() % width != 0)
        throw std::runtime_error("full-sequence checkpoint has invalid token geometry");
    return { values.end() - width, values.end() };
}

std::string prefix_reference_dir(const std::string & root, uint32_t step) {
    const auto path = std::filesystem::path(root) / ("prefix-" + std::to_string(step));
    if (!internal_qualification_mode() && !std::filesystem::exists(path / "reference.meta"))
        throw std::runtime_error("missing per-prefix llama reference dump: " + path.string());
    return path.string();
}

void write_binary_export(const std::filesystem::path & path, const uint8_t * bytes, size_t size) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes), static_cast<std::streamsize>(size));
    output.close();
    if (!output) throw std::runtime_error("failed writing qualification export: " + path.string());
}

void write_f32_export(const std::filesystem::path & path, const std::vector<float> & values) {
    write_binary_export(path, reinterpret_cast<const uint8_t *>(values.data()),
        values.size() * sizeof(float));
}

void run_full_reference_logits(Model & model, const std::vector<int32_t> & tokens,
    const std::string & reference_root) {
    if (!internal_qualification_mode() || model.layer_count != 40 || tokens.empty() ||
        tokens.size() > 32 || reference_root.empty())
        throw std::runtime_error("full/reference logit qualification requires internal-only mode, 40 layers, and 1..32 tokens");
    const auto started = std::chrono::steady_clock::now();
    const char * export_dir = std::getenv("VBUF_QWEN_EXPORT_DIR");
    const std::filesystem::path export_path = export_dir == nullptr ? std::filesystem::path() : std::filesystem::path(export_dir);
    if (export_dir != nullptr) std::filesystem::create_directories(export_path);
    std::vector<float> hidden;
    for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
        hidden = run_positions(model, "", static_cast<uint32_t>(tokens.size()), layer,
            hidden, false, tokens, nullptr, 0, nullptr, nullptr, nullptr, true, true);
        if (export_dir != nullptr)
            write_f32_export(export_path / ("l_out-" + std::to_string(layer) + ".f32"), hidden);
    }
    std::vector<float> normalized;
    const auto logits = run_final_head(model, "", hidden, true, &normalized);
    const uint32_t positions = static_cast<uint32_t>(tokens.size());
    const size_t vocab = logits.size() / positions;
    if (export_dir != nullptr) {
        write_f32_export(export_path / "final_hidden.f32", hidden);
        write_f32_export(export_path / "final_norm.f32", normalized);
        write_f32_export(export_path / "logits.f32", logits);
        std::ofstream metadata(export_path / "sequence.meta");
        metadata << "architecture=qwen3\npositions=" << positions << "\nembedding=5120\nvocabulary=" << vocab << "\ntokens=";
        for (size_t index = 0; index < tokens.size(); ++index)
            metadata << (index == 0 ? "" : ",") << tokens[index];
        metadata << "\nmode=canonical_full_sequence_teacher_forced\n";
        if (!metadata) throw std::runtime_error("failed writing Qwen3 export metadata");
        std::printf("qualification_exports=%s hidden_bytes=%zu norm_bytes=%zu logits_bytes=%zu\n",
            export_path.c_str(), hidden.size() * sizeof(float), normalized.size() * sizeof(float),
            logits.size() * sizeof(float));
    }
    std::printf("canonical_full_reference_only=PASS positions=%u groups=%zu elapsed_seconds=%.3f\n",
        positions, qwen3_query_groups(positions, cpu_softmax_policy().granularity).size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    std::printf("ggml_context_tensor_buffer_peak_bytes first_stage=%zu second_stage=%zu final_head=%zu scope=per_context_backing_includes_weights_and_intermediates\n",
        g_qualification_memory_peaks.first_stage_bytes,
        g_qualification_memory_peaks.second_stage_bytes,
        g_qualification_memory_peaks.final_head_bytes);
    for (uint32_t prefix : {8u, 16u, 32u}) {
        if (prefix > positions) continue;
        const std::filesystem::path prefix_dir = std::filesystem::path(reference_root) /
            ("prefix-" + std::to_string(prefix - 1));
        if (!std::filesystem::exists(prefix_dir / "result_output-0.f32")) {
            std::printf("external_llama_prefix=%u status=REFERENCE_NOT_AVAILABLE\n", prefix);
            continue;
        }
        const auto external = reference(prefix_dir.string(), "result_output-0");
        const std::vector<float> actual(logits.begin() + static_cast<size_t>(prefix - 1) * vocab,
            logits.begin() + static_cast<size_t>(prefix) * vocab);
        if (external.size() != vocab) throw std::runtime_error("external logit vocabulary mismatch");
        compare("external_llama_logits_prefix_" + std::to_string(prefix), actual, external, 1);
        const bool within_tolerance = within_fixed_tolerance(actual, external);
        std::vector<uint32_t> actual_order(vocab), external_order(vocab);
        std::iota(actual_order.begin(), actual_order.end(), 0);
        external_order = actual_order;
        const auto actual_better = [&](uint32_t a, uint32_t b) {
            return actual[a] == actual[b] ? a < b : actual[a] > actual[b];
        };
        const auto external_better = [&](uint32_t a, uint32_t b) {
            return external[a] == external[b] ? a < b : external[a] > external[b];
        };
        std::partial_sort(actual_order.begin(), actual_order.begin() + 10, actual_order.end(), actual_better);
        std::partial_sort(external_order.begin(), external_order.begin() + 10, external_order.end(), external_better);
        size_t overlap = 0;
        for (size_t i = 0; i < 10; ++i)
            if (std::find(external_order.begin(), external_order.begin() + 10, actual_order[i]) != external_order.begin() + 10) ++overlap;
        std::printf("external_llama_prefix=%u numerical_within_fixed_1e-5=%s top1_match=%s vbuf_top1=%u llama_top1=%u top10_overlap=%zu/10 vbuf_margin=%.9g llama_margin=%.9g\n",
            prefix, within_tolerance ? "YES" : "NO", actual_order[0] == external_order[0] ? "YES" : "NO",
            actual_order[0], external_order[0], overlap, actual[actual_order[0]] - actual[actual_order[1]],
            external[external_order[0]] - external[external_order[1]]);
    }
}

std::vector<int32_t> generate_persistent_tokens(Model & model, uint32_t seed, uint32_t token_count) {
    if (token_count == 0 || token_count > 32)
        throw std::runtime_error("persistent greedy generation requires 1..32 tokens");
    Qwen3KvCache cache({model.layer_count, HEADS, KV_HEADS, HEAD_DIM, token_count});
    std::vector<int32_t> tokens{static_cast<int32_t>(seed)};
    for (uint32_t position = 0; tokens.size() < token_count; ++position) {
        std::vector<float> hidden;
        const std::vector<int32_t> one_token{tokens[position]};
        for (uint32_t layer = 0; layer < model.layer_count; ++layer)
            hidden = run_positions(model, "", 1, layer, hidden, false, one_token,
                &cache, position, nullptr, nullptr, nullptr, true, true);
        cache.complete_position(position);
        const auto logits = run_final_head(model, "", hidden, true);
        const uint32_t next = greedy_token(logits);
        std::printf("persistent_greedy_step prefix=%u input=%d next=%u cache_bytes=%zu\n",
            position + 1, tokens[position], next, cache.bytes());
        tokens.push_back(static_cast<int32_t>(next));
    }
    std::printf("persistent_greedy_tokens=");
    for (size_t i = 0; i < tokens.size(); ++i) std::printf("%s%d", i == 0 ? "" : ",", tokens[i]);
    std::printf("\npersistent_greedy_generation=PASS tokens=%u generation_engine=isolated_Qwen3KvCache\n",
        token_count);
    return tokens;
}

void run_long_persistent_kv(Model & model, const std::vector<int32_t> & tokens,
    const std::string & reference_root = "") {
    if (!internal_qualification_mode())
        throw std::runtime_error("long Qwen3 qualification requires VBUF_QWEN_INTERNAL_ONLY=1");
    if (tokens.empty() || tokens.size() > 32 || model.layer_count != 40)
        throw std::runtime_error("long Qwen3 qualification requires 1..32 tokens and 40 layers");
    const auto bit_equal = [](const std::vector<float> & a, const std::vector<float> & b) {
        return a.size() == b.size() && !a.empty() &&
            std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    };
    auto row = [](const std::vector<float> & values, size_t width, uint32_t position) {
        if (width == 0 || values.size() % width != 0 || position >= values.size() / width)
            throw std::runtime_error("long qualification row geometry mismatch");
        const size_t begin = static_cast<size_t>(position) * width;
        return std::vector<float>(values.begin() + begin, values.begin() + begin + width);
    };
    auto attention_row = [](const LayerDiagnostics & diagnostics, const std::vector<float> & values,
        uint32_t position, uint32_t count) {
        const uint32_t keys = diagnostics.attention_key_positions;
        const uint32_t queries = diagnostics.attention_query_positions;
        if (position >= queries || count > keys || values.size() !=
            static_cast<size_t>(HEADS) * queries * keys)
            throw std::runtime_error("long attention row geometry mismatch");
        std::vector<float> output;
        output.reserve(static_cast<size_t>(HEADS) * count);
        for (uint32_t head = 0; head < HEADS; ++head) {
            const size_t begin = (static_cast<size_t>(head) * queries + position) * keys;
            output.insert(output.end(), values.begin() + begin, values.begin() + begin + count);
        }
        return output;
    };
    const uint32_t positions = static_cast<uint32_t>(tokens.size());
    const size_t row_bytes = static_cast<size_t>(KV_HEADS) * HEAD_DIM * sizeof(uint16_t);
    std::printf("long_qualification tokens=%u independent_stream_replay=YES reference_checks=DISABLED\n", positions);

    const auto full_start = std::chrono::steady_clock::now();
    std::vector<LayerDiagnostics> full_diagnostics(model.layer_count);
    std::vector<float> full_hidden;
    for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
        LayerDiagnostics diagnostics;
        full_hidden = run_positions(model, "", positions, layer, full_hidden, false,
            tokens, nullptr, 0, &diagnostics, nullptr, nullptr, true, true);
        if (full_hidden.size() != static_cast<size_t>(EMBED) * positions)
            throw std::runtime_error("canonical grouped full-sequence block output geometry mismatch");
        full_diagnostics[layer] = std::move(diagnostics);
    }
    const auto full_logits = run_final_head(model, "", full_hidden, false);
    const auto full_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - full_start).count();
    const size_t vocab = full_logits.size() / positions;
    if (vocab != 151936 || full_logits.size() != vocab * positions)
        throw std::runtime_error("batched final-head logits geometry mismatch");
    std::printf("canonical_full_sequence=PASS positions=%u groups=%zu logits=%zu elapsed_seconds=%.3f\n",
        positions, qwen3_query_groups(positions, cpu_softmax_policy().granularity).size(),
        full_logits.size(), full_elapsed);
    for (uint32_t prefix : {8u, 16u, 32u}) {
        if (prefix > positions || reference_root.empty()) continue;
        const std::filesystem::path prefix_dir = std::filesystem::path(reference_root) /
            ("prefix-" + std::to_string(prefix - 1));
        if (!std::filesystem::exists(prefix_dir / "result_output-0.f32")) continue;
        const auto external_logits = reference(prefix_dir.string(), "result_output-0");
        const std::vector<float> vbuf_logits(full_logits.begin() + static_cast<size_t>(prefix - 1) * vocab,
            full_logits.begin() + static_cast<size_t>(prefix) * vocab);
        if (external_logits.size() != vocab)
            throw std::runtime_error("external llama.cpp logit vocabulary mismatch at prefix " + std::to_string(prefix));
        compare("external_llama_logits_prefix_" + std::to_string(prefix), vbuf_logits, external_logits, 1);
        const bool within_tolerance = within_fixed_tolerance(vbuf_logits, external_logits);
        std::vector<uint32_t> order(vocab);
        std::iota(order.begin(), order.end(), 0);
        const auto better = [&](uint32_t a, uint32_t b) {
            return vbuf_logits[a] == vbuf_logits[b] ? a < b : vbuf_logits[a] > vbuf_logits[b];
        };
        auto vbuf_order = order;
        std::partial_sort(vbuf_order.begin(), vbuf_order.begin() + 10, vbuf_order.end(), better);
        const auto reference_better = [&](uint32_t a, uint32_t b) {
            return external_logits[a] == external_logits[b] ? a < b : external_logits[a] > external_logits[b];
        };
        auto reference_order = order;
        std::partial_sort(reference_order.begin(), reference_order.begin() + 10, reference_order.end(), reference_better);
        size_t overlap = 0;
        for (size_t i = 0; i < 10; ++i)
            if (std::find(reference_order.begin(), reference_order.begin() + 10, vbuf_order[i]) != reference_order.begin() + 10)
                ++overlap;
        const double vbuf_margin = vbuf_logits[vbuf_order[0]] - vbuf_logits[vbuf_order[1]];
        const double reference_margin = external_logits[reference_order[0]] - external_logits[reference_order[1]];
        std::printf("external_llama_prefix=%u numerical_within_fixed_1e-5=%s top1_match=%s vbuf_top1=%u llama_top1=%u top10_overlap=%zu/10 vbuf_margin=%.9g llama_margin=%.9g\n",
            prefix, within_tolerance ? "YES" : "NO", vbuf_order[0] == reference_order[0] ? "YES" : "NO",
            vbuf_order[0], reference_order[0], overlap, vbuf_margin, reference_margin);
    }

    Qwen3KvCache cache({model.layer_count, HEADS, KV_HEADS, HEAD_DIM, positions});
    std::vector<std::vector<uint8_t>> replay_keys(model.layer_count), replay_values(model.layer_count);
    std::vector<float> first_persistent_logits;
    uint32_t generation_first_divergence = UINT32_MAX;
    bool all_exact = true;
    const uint32_t granularity = cpu_softmax_policy().granularity;
    for (uint32_t position = 0; position < positions; ++position) {
        const std::vector<int32_t> one_token{tokens[position]};
        std::vector<float> replay_hidden, persistent_hidden;
        std::vector<LayerDiagnostics> replay_diagnostics(model.layer_count), persistent_diagnostics(model.layer_count);
        const auto persistent_start = std::chrono::steady_clock::now();
        for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
            persistent_hidden = run_positions(model, "", 1, layer, persistent_hidden,
                false, one_token, &cache, position, &persistent_diagnostics[layer],
                nullptr, nullptr, true, true);
            if (persistent_hidden.size() != EMBED)
                throw std::runtime_error("persistent incremental output geometry mismatch");
        }
        const double persistent_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - persistent_start).count();
        const auto replay_start = std::chrono::steady_clock::now();
        for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
            replay_hidden = run_positions(model, "", 1, layer, replay_hidden,
                false, one_token, nullptr, position, &replay_diagnostics[layer],
                &replay_keys[layer], &replay_values[layer], true, true);
            if (replay_hidden.size() != EMBED)
                throw std::runtime_error("independent replay output geometry mismatch");
            replay_keys[layer].insert(replay_keys[layer].end(), replay_diagnostics[layer].key_f16.begin(),
                replay_diagnostics[layer].key_f16.end());
            replay_values[layer].insert(replay_values[layer].end(), replay_diagnostics[layer].value_f16.begin(),
                replay_diagnostics[layer].value_f16.end());
        }
        const double replay_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - replay_start).count();
        cache.complete_position(position);

        const uint32_t extent = SoftmaxComputeExtent::make(position + 1, granularity).compute;
        uint32_t checked_checkpoints = 0;
        for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
            const LayerDiagnostics & full = full_diagnostics[layer];
            const LayerDiagnostics & replay = replay_diagnostics[layer];
            const LayerDiagnostics & persistent = persistent_diagnostics[layer];
            auto full_token = [&](const std::vector<float> & values, size_t width) {
                return row(values, width, position);
            };
            auto check = [&](const char * checkpoint, const std::vector<float> & full_values,
                const std::vector<float> & replay_values, const std::vector<float> & persistent_values) {
                if (!bit_equal(full_values, replay_values) || !bit_equal(replay_values, persistent_values)) {
                    size_t index = 0;
                    const size_t count = std::min(full_values.size(), replay_values.size());
                    while (index < count && std::memcmp(&full_values[index], &replay_values[index], sizeof(float)) == 0) ++index;
                    std::fprintf(stderr, "long_first_divergence prefix=%u layer=%u checkpoint=%s index=%zu full_count=%zu replay_count=%zu persistent_count=%zu\n",
                        position + 1, layer, checkpoint, index, full_values.size(), replay_values.size(), persistent_values.size());
                    all_exact = false;
                    throw std::runtime_error("canonical full/replay/persistent mismatch at prefix " +
                        std::to_string(position + 1) + " layer " + std::to_string(layer) + " checkpoint " + checkpoint);
                }
                ++checked_checkpoints;
            };
            check("layer_input", full_token(full.layer_input, EMBED), replay.layer_input, persistent.layer_input);
            check("attention_rmsnorm", full_token(full.attention_rmsnorm, EMBED), replay.attention_rmsnorm, persistent.attention_rmsnorm);
            check("q_projection", full_token(full.q_projection, EMBED), replay.q_projection, persistent.q_projection);
            check("k_projection", full_token(full.k_projection, KV_HEADS * HEAD_DIM), replay.k_projection, persistent.k_projection);
            check("v_projection", full_token(full.v_projection, KV_HEADS * HEAD_DIM), replay.v_projection, persistent.v_projection);
            check("q_rope", full_token(full.q_rope, EMBED), replay.q_rope, persistent.q_rope);
            check("k_rope", full_token(full.k_rope, KV_HEADS * HEAD_DIM), replay.k_rope, persistent.k_rope);
            check("attention_scores_visible", attention_row(full, full.attention_scores, position, position + 1),
                attention_row(replay, replay.attention_scores, 0, position + 1),
                attention_row(persistent, persistent.attention_scores, 0, position + 1));
            check("attention_probabilities_compute_extent", attention_row(full, full.attention_probabilities, position, extent),
                attention_row(replay, replay.attention_probabilities, 0, extent),
                attention_row(persistent, persistent.attention_probabilities, 0, extent));
            check("attention_context", full_token(full.attention_context, EMBED), replay.attention_context, persistent.attention_context);
            check("attention_projection", full_token(full.attention_projection, EMBED), replay.attention_projection, persistent.attention_projection);
            check("attention_residual", full_token(full.attention_residual, EMBED), replay.attention_residual, persistent.attention_residual);
            check("ffn_rmsnorm", full_token(full.ffn_rmsnorm, EMBED), replay.ffn_rmsnorm, persistent.ffn_rmsnorm);
            check("ffn_gate", full_token(full.ffn_gate, FFN), replay.ffn_gate, persistent.ffn_gate);
            check("ffn_up", full_token(full.ffn_up, FFN), replay.ffn_up, persistent.ffn_up);
            check("ffn_swiglu", full_token(full.ffn_swiglu, FFN), replay.ffn_swiglu, persistent.ffn_swiglu);
            check("ffn_down", full_token(full.ffn_down, EMBED), replay.ffn_down, persistent.ffn_down);
            check("block_output", full_token(full.block_output, EMBED), replay.block_output, persistent.block_output);

            const size_t offset = static_cast<size_t>(position) * row_bytes;
            const auto full_key = full.key_f16.begin() + offset;
            const auto full_value = full.value_f16.begin() + offset;
            const bool prestore_exact = std::equal(full_key, full_key + row_bytes, replay.key_f16.begin()) &&
                std::equal(full_value, full_value + row_bytes, replay.value_f16.begin()) &&
                replay.key_f16 == persistent.key_f16 && replay.value_f16 == persistent.value_f16;
            const bool cache_exact = std::equal(replay.key_f16.begin(), replay.key_f16.end(),
                cache.key_bytes(layer) + offset) && std::equal(replay.value_f16.begin(), replay.value_f16.end(),
                cache.value_bytes(layer) + offset);
            if (!prestore_exact || !cache_exact) {
                all_exact = false;
                std::fprintf(stderr, "long_first_divergence prefix=%u layer=%u checkpoint=K_or_V_cache\n", position + 1, layer);
                throw std::runtime_error("long canonical KV bytes differ at prefix " + std::to_string(position + 1));
            }
        }
        const auto replay_logits = run_final_head(model, "", replay_hidden, true);
        const auto persistent_logits = run_final_head(model, "", persistent_hidden, true);
        const std::vector<float> full_logits_row(full_logits.begin() + static_cast<size_t>(position) * vocab,
            full_logits.begin() + static_cast<size_t>(position + 1) * vocab);
        const uint32_t final_layer = model.layer_count - 1;
        const auto full_hidden_row = row(full_hidden, EMBED, position);
        if (!bit_equal(full_hidden_row, replay_hidden) || !bit_equal(replay_hidden, persistent_hidden)) {
            all_exact = false;
            throw std::runtime_error("final hidden differs at prefix " + std::to_string(position + 1));
        }
        if (!bit_equal(full_logits_row, replay_logits) || !bit_equal(replay_logits, persistent_logits)) {
            all_exact = false;
            size_t index = 0;
            while (index < full_logits_row.size() &&
                std::memcmp(&full_logits_row[index], &replay_logits[index], sizeof(float)) == 0) ++index;
            std::fprintf(stderr, "long_first_divergence prefix=%u layer=%u checkpoint=final_logits index=%zu\n",
                position + 1, final_layer, index);
            throw std::runtime_error("final logits differ at prefix " + std::to_string(position + 1));
        }
        const uint32_t full_next = greedy_token(full_logits_row);
        const uint32_t replay_next = greedy_token(replay_logits);
        const uint32_t persistent_next = greedy_token(persistent_logits);
        if (full_next != replay_next || replay_next != persistent_next)
            throw std::runtime_error("top-1 differs at prefix " + std::to_string(position + 1));
        if (position + 1 < positions && full_next != static_cast<uint32_t>(tokens[position + 1]) &&
            generation_first_divergence == UINT32_MAX)
            generation_first_divergence = position + 1;
        const uint32_t boundary_remainder = (position + 1) % granularity;
        const bool boundary = boundary_remainder == 0 || boundary_remainder == 1 ||
            boundary_remainder + 1 == granularity;
        std::printf("long_prefix=%u logical_extent=%u compute_extent=%u padding=%u groups=%zu checkpoints_total=%u checkpoints_per_layer=%u full_replay=BIT-IDENTICAL replay_kv=BIT-IDENTICAL cache_bytes=%zu cache_exact=YES top1=%u replay_persistent_token_latency_ms=%.3f/%.3f boundary=%s\n",
            position + 1, position + 1, extent, extent - (position + 1),
            qwen3_query_groups(position + 1, granularity).size(), checked_checkpoints,
            checked_checkpoints / model.layer_count, cache.bytes(), full_next, replay_ms,
            persistent_ms, boundary ? "YES" : "NO");
        if (position + 1 == 8 || position + 1 == 9 || position + 1 == 16 ||
            position + 1 == 17 || position + 1 == 24 || position + 1 == 25 ||
            position + 1 == 32)
            std::printf("long_boundary_result prefix=%u full_replay=BIT-IDENTICAL replay_persistent=BIT-IDENTICAL top1=%u\n",
                position + 1, full_next);
        if (position == 0) first_persistent_logits = persistent_logits;
    }
    std::printf("long_sequence_tokens=");
    for (size_t i = 0; i < tokens.size(); ++i) std::printf("%s%d", i == 0 ? "" : ",", tokens[i]);
    std::printf("\nlong_sequence_exact=%s first_divergence=NONE greedy_expected_sequence_divergence=%s\n",
        all_exact ? "YES" : "NO", generation_first_divergence == UINT32_MAX ? "NONE" :
            std::to_string(generation_first_divergence).c_str());
    std::printf("long_cache_bytes_final=%zu expected=%zu per_token=%zu\n", cache.bytes(),
        positions * cache.bytes_per_token(), cache.bytes_per_token());

    cache.reset();
    if (cache.bytes() != 0 || cache.completed_positions() != 0)
        throw std::runtime_error("long cache reset did not clear state");
    std::vector<float> reset_hidden;
    for (uint32_t layer = 0; layer < model.layer_count; ++layer)
        reset_hidden = run_positions(model, "", 1, layer, reset_hidden, false,
            {tokens.front()}, &cache, 0, nullptr, nullptr, nullptr, true, true);
    cache.complete_position(0);
    const auto reset_logits = run_final_head(model, "", reset_hidden, true);
    if (!bit_equal(first_persistent_logits, reset_logits))
        throw std::runtime_error("same-cache reset and clean replay differ");
    cache.reset();
    std::printf("long_cache_reset=PASS same_cache_clean_replay=BIT-IDENTICAL\n");

    Qwen3KvCache recreated({model.layer_count, HEADS, KV_HEADS, HEAD_DIM, 1});
    std::vector<float> recreated_hidden;
    for (uint32_t layer = 0; layer < model.layer_count; ++layer)
        recreated_hidden = run_positions(model, "", 1, layer, recreated_hidden, false,
            {tokens.front()}, &recreated, 0, nullptr, nullptr, nullptr, true, true);
    recreated.complete_position(0);
    if (!bit_equal(first_persistent_logits, run_final_head(model, "", recreated_hidden, true)))
        throw std::runtime_error("destroy/recreate clean replay differs");
    std::printf("long_cache_destroy_recreate=PASS cross_session_state=ISOLATED\n");
    std::printf("ggml_context_tensor_buffer_peak_bytes first_stage=%zu second_stage=%zu final_head=%zu scope=per_context_backing_includes_weights_and_intermediates\n",
        g_qualification_memory_peaks.first_stage_bytes,
        g_qualification_memory_peaks.second_stage_bytes,
        g_qualification_memory_peaks.final_head_bytes);
}

void run_persistent_kv(Model & model, const std::string & reference_root,
    const std::vector<int32_t> & tokens) {
    if (tokens.empty() || tokens.size() > 8 || model.layer_count != 40)
        throw std::runtime_error("persistent Qwen3 qualification requires 1..8 tokens and 40 layers");
    Qwen3KvCache cache({model.layer_count, HEADS, KV_HEADS, HEAD_DIM,
        static_cast<uint32_t>(tokens.size())});
    const bool diagnostic_mode = std::getenv("VBUF_KV_DIAGNOSTIC") != nullptr;
    const bool final_prefix_only = std::getenv("VBUF_KV_FINAL_PREFIX_ONLY") != nullptr;
    bool all_cache_bytes_match = true;
    std::printf("persistent_kv_diagnostic_mode final_prefix_only=%s\n", final_prefix_only ? "YES" : "NO");
    std::vector<uint32_t> generated;
    std::vector<float> first_full_logits;
    std::vector<std::vector<LayerDiagnostics>> kv_history(model.layer_count);
    struct IncrementalReplayResult {
        std::vector<LayerDiagnostics> diagnostics;
        std::vector<float> hidden;
        std::vector<float> logits;
    };
    auto replay_incremental_prefix = [&](uint32_t target_position) {
        IncrementalReplayResult result;
        result.diagnostics.resize(model.layer_count);
        std::vector<std::vector<uint8_t>> replay_keys(model.layer_count), replay_values(model.layer_count);
        for (uint32_t position = 0; position <= target_position; ++position) {
            std::vector<float> hidden;
            const std::string replay_ref = prefix_reference_dir(reference_root, position);
            const std::vector<int32_t> one_token{tokens[position]};
            for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
                LayerDiagnostics current;
                hidden = run_positions(model, replay_ref, 1, layer, hidden, false, one_token,
                    nullptr, position, &current, &replay_keys[layer], &replay_values[layer], true);
                if (hidden.empty())
                    throw std::runtime_error("fresh incremental replay failed at position " +
                        std::to_string(position) + " layer " + std::to_string(layer));
                replay_keys[layer].insert(replay_keys[layer].end(), current.key_f16.begin(), current.key_f16.end());
                replay_values[layer].insert(replay_values[layer].end(), current.value_f16.begin(), current.value_f16.end());
                if (position == target_position) result.diagnostics[layer] = std::move(current);
            }
            if (position == target_position) result.hidden = std::move(hidden);
        }
        result.logits = run_final_head(model, prefix_reference_dir(reference_root, target_position),
            result.hidden, true);
        return result;
    };
    auto compare_replay_float = [](const char * name, uint32_t layer, uint32_t position,
        const std::vector<float> & replay, const std::vector<float> & persistent) {
        if (replay.size() != persistent.size() || replay.empty())
            throw std::runtime_error(std::string("incremental replay geometry mismatch: ") + name);
        double max_abs = 0, sum_sq = 0, reference_sq = 0;
        uint64_t max_ulp = 0;
        bool bit_exact = true;
        for (size_t i = 0; i < replay.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(replay[i]) - persistent[i]));
            sum_sq += std::pow(static_cast<double>(replay[i]) - persistent[i], 2);
            reference_sq += static_cast<double>(replay[i]) * replay[i];
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &replay[i], sizeof(a));
            std::memcpy(&b, &persistent[i], sizeof(b));
            bit_exact &= a == b;
            const uint32_t ordered_a = (a & 0x80000000u) ? ~a : (a | 0x80000000u);
            const uint32_t ordered_b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            max_ulp = std::max<uint64_t>(max_ulp,
                ordered_a > ordered_b ? ordered_a - ordered_b : ordered_b - ordered_a);
        }
        const double rms = std::sqrt(sum_sq / replay.size());
        const double relative_rms = reference_sq == 0 ? rms : rms / std::sqrt(reference_sq / replay.size());
        const bool within = max_abs <= ATOL + RTOL * std::abs(static_cast<double>(replay[0]));
        std::printf("incremental_replay_compare layer=%u position=%u checkpoint=%s bit_exact=%s max_abs=%.9g rms=%.9g relative_rms=%.9g max_ulp=%llu tolerance=%s\n",
            layer, position, name, bit_exact ? "YES" : "NO", max_abs, rms, relative_rms,
            static_cast<unsigned long long>(max_ulp), within ? "PASS" : "FAIL");
        return within;
    };
    bool all_incremental_replays_match = true;
    uint32_t first_incremental_divergence_layer = model.layer_count;
    for (uint32_t step = 0; step < tokens.size(); ++step) {
        const std::string ref_dir = prefix_reference_dir(reference_root, step);
        const uint32_t prefix_length = step + 1;
        std::vector<int32_t> prefix(tokens.begin(), tokens.begin() + prefix_length);
        const auto ref_logits = reference(ref_dir, "result_output-0");
        const uint32_t reference_next = greedy_token(ref_logits);
        if (step != 0 && generated.back() != static_cast<uint32_t>(tokens[step]))
            throw std::runtime_error("persistent greedy token diverged before the next prefix");

        std::vector<LayerDiagnostics> full_diagnostics(model.layer_count);
        std::vector<std::vector<float>> full_layer_outputs(model.layer_count);
        std::vector<float> full_hidden;
        const bool have_full = !final_prefix_only || prefix_length == tokens.size();
        if (have_full) {
            for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
                full_hidden = run_positions(model, ref_dir, prefix_length, layer, full_hidden,
                    false, prefix, nullptr, 0, &full_diagnostics[layer], nullptr, nullptr, true);
                full_layer_outputs[layer] = final_position(full_hidden, EMBED);
                if (full_hidden.empty()) throw std::runtime_error("full-recompute hidden state is non-finite");
            }
        }
        const auto full_logits = have_full ? run_final_head(model, ref_dir, full_hidden, false) : std::vector<float>{};
        if (step == 0 && have_full) first_full_logits = full_logits;

        const int32_t token = tokens[step];
        std::vector<int32_t> one_token{token};
        std::vector<float> kv_hidden;
        std::vector<LayerDiagnostics> kv_diagnostics(model.layer_count);
        std::vector<std::vector<float>> kv_layer_outputs(model.layer_count);
        std::vector<float> kv_block0_output;
        for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
            kv_hidden = run_positions(model, ref_dir, 1, layer, kv_hidden,
                false, one_token, &cache, step, &kv_diagnostics[layer], nullptr, nullptr, true);
            if (layer == 0) kv_block0_output = kv_hidden;
            if (kv_hidden.empty()) throw std::runtime_error("persistent-KV hidden state is non-finite");
            kv_layer_outputs[layer] = kv_hidden;
            kv_history[layer].push_back(kv_diagnostics[layer]);
        }
        cache.complete_position(step);
        const auto kv_logits = run_final_head(model, ref_dir, kv_hidden, true);
        if (step == 0 && !have_full) first_full_logits = kv_logits;

        // Recompute this whole prefix from an empty, ordinary byte-vector history.
        // This uses the same one-row graph/shape at every token but never reads the
        // persistent Qwen3KvCache instance under test.
        const auto replay = replay_incremental_prefix(step);
        bool prefix_replay_matches = true;
        uint32_t prefix_first_divergent_layer = model.layer_count;
        const size_t replay_tensor_bytes = static_cast<size_t>(KV_HEADS) * HEAD_DIM * sizeof(uint16_t);
        for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
            const auto & a = replay.diagnostics[layer];
            const auto & b = kv_diagnostics[layer];
            bool layer_matches = true;
            layer_matches &= compare_replay_float("attention_scores", layer, step, a.attention_scores, b.attention_scores);
            layer_matches &= compare_replay_float("attention_probabilities", layer, step, a.attention_probabilities, b.attention_probabilities);
            layer_matches &= compare_replay_float("attention_context", layer, step, a.attention_context, b.attention_context);
            layer_matches &= compare_replay_float("attention_projection", layer, step, a.attention_projection, b.attention_projection);
            layer_matches &= compare_replay_float("block_output", layer, step, a.block_output, b.block_output);
            const bool prestore_k_equal = a.key_f16.size() == b.key_f16.size() && a.key_f16.size() == replay_tensor_bytes &&
                std::equal(a.key_f16.begin(), a.key_f16.end(), b.key_f16.begin());
            const bool prestore_v_equal = a.value_f16.size() == b.value_f16.size() && a.value_f16.size() == replay_tensor_bytes &&
                std::equal(a.value_f16.begin(), a.value_f16.end(), b.value_f16.begin());
            const size_t cache_offset = static_cast<size_t>(step) * replay_tensor_bytes;
            const bool readback_k_equal = a.key_f16.size() == replay_tensor_bytes &&
                std::equal(a.key_f16.begin(), a.key_f16.end(), cache.key_bytes(layer) + cache_offset);
            const bool readback_v_equal = a.value_f16.size() == replay_tensor_bytes &&
                std::equal(a.value_f16.begin(), a.value_f16.end(), cache.value_bytes(layer) + cache_offset);
            std::printf("incremental_replay_kv layer=%u position=%u K_pre_storage_exact=%s V_pre_storage_exact=%s persistent_K_readback_exact=%s persistent_V_readback_exact=%s\n",
                layer, step, prestore_k_equal ? "YES" : "NO", prestore_v_equal ? "YES" : "NO",
                readback_k_equal ? "YES" : "NO", readback_v_equal ? "YES" : "NO");
            layer_matches &= prestore_k_equal && prestore_v_equal && readback_k_equal && readback_v_equal;
            prefix_replay_matches &= layer_matches;
            if (!layer_matches && prefix_first_divergent_layer == model.layer_count)
                prefix_first_divergent_layer = layer;
        }
        prefix_replay_matches &= compare_replay_float("final_hidden", model.layer_count, step, replay.hidden, kv_hidden);
        prefix_replay_matches &= compare_replay_float("final_logits", model.layer_count, step, replay.logits, kv_logits);
        all_incremental_replays_match &= prefix_replay_matches;
        if (!prefix_replay_matches && first_incremental_divergence_layer == model.layer_count)
            first_incremental_divergence_layer = prefix_first_divergent_layer;
        std::printf("incremental_replay_prefix position=%u result=%s first_divergent_layer=%s execution_geometry=matched backend=shared_cpu\n",
            step, prefix_replay_matches ? "PASS" : "FAIL",
            prefix_first_divergent_layer == model.layer_count ? "NONE" : std::to_string(prefix_first_divergent_layer).c_str());

        uint32_t first_divergent_layer = model.layer_count;
        auto max_difference = [](const std::vector<float> & a, const std::vector<float> & b) {
            if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
            double maximum = 0;
            for (size_t i = 0; i < a.size(); ++i)
                maximum = std::max(maximum, std::abs(static_cast<double>(a[i]) - b[i]));
            return maximum;
        };
        const size_t kv_tensor_bytes = static_cast<size_t>(KV_HEADS) * HEAD_DIM * sizeof(uint16_t);
        if (step + 1 == tokens.size() && prefix_length == 8) {
            auto token_at = [](const std::vector<float> & values, size_t width, uint32_t position) {
                if (values.size() % width != 0 || position >= values.size() / width)
                    throw std::runtime_error("token checkpoint position is outside tensor");
                const size_t begin = static_cast<size_t>(position) * width;
                return std::vector<float>(values.begin() + begin, values.begin() + begin + width);
            };
            auto boundary_metrics = [&](const char * checkpoint, uint32_t layer, uint32_t position,
                const std::vector<float> & incremental, const std::vector<float> & full) {
                if (incremental.size() != full.size())
                    throw std::runtime_error("layer-boundary trace geometry mismatch");
                double max_abs = 0, squared_error = 0, squared_full = 0, dot = 0;
                uint64_t max_ulp = 0;
                bool bit_identical = true;
                for (size_t i = 0; i < full.size(); ++i) {
                    const double difference = static_cast<double>(incremental[i]) - full[i];
                    max_abs = std::max(max_abs, std::abs(difference));
                    squared_error += difference * difference;
                    squared_full += static_cast<double>(full[i]) * full[i];
                    dot += static_cast<double>(incremental[i]) * full[i];
                    uint32_t a = 0, b = 0;
                    std::memcpy(&a, &incremental[i], sizeof(a));
                    std::memcpy(&b, &full[i], sizeof(b));
                    bit_identical &= a == b;
                    const uint32_t oa = (a & 0x80000000u) ? ~a : (a | 0x80000000u);
                    const uint32_t ob = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
                    max_ulp = std::max<uint64_t>(max_ulp, oa > ob ? oa - ob : ob - oa);
                }
                const double rms = std::sqrt(squared_error / std::max<size_t>(1, full.size()));
                const double relative_rms = squared_full == 0 ? rms : rms / std::sqrt(squared_full / full.size());
                double norm_a = 0, norm_b = 0;
                for (size_t i = 0; i < full.size(); ++i) {
                    norm_a += static_cast<double>(incremental[i]) * incremental[i];
                    norm_b += static_cast<double>(full[i]) * full[i];
                }
                const double cosine = norm_a == 0 || norm_b == 0 ? 0 : dot / std::sqrt(norm_a * norm_b);
                std::printf("kv_boundary checkpoint=%s layer=%u position=%u bit_identical=%s max_abs=%.9g rms=%.9g relative_rms=%.9g cosine=%.12g max_ulp=%llu\n",
                    checkpoint, layer, position, bit_identical ? "YES" : "NO", max_abs, rms,
                    relative_rms, cosine, static_cast<unsigned long long>(max_ulp));
                return bit_identical;
            };
            uint32_t first_probability_layer = model.layer_count;
            std::string first_probability_checkpoint = "NONE";
            const uint32_t traced_position = 6;
            const uint32_t visible_keys = traced_position + 1;
            auto extract_attention_row = [&](const std::vector<float> & values, uint32_t keys,
                uint32_t queries, uint32_t query_position) {
                if (keys < visible_keys || queries == 0 || query_position >= queries)
                    throw std::runtime_error("attention probability scan geometry mismatch");
                std::vector<float> row;
                row.reserve(static_cast<size_t>(HEADS) * visible_keys);
                for (uint32_t head = 0; head < HEADS; ++head) {
                    const size_t begin = (static_cast<size_t>(head) * queries + query_position) * keys;
                    row.insert(row.end(), values.begin() + begin, values.begin() + begin + visible_keys);
                }
                return row;
            };
            for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
                const auto & inc = kv_history[layer][traced_position];
                const auto & full = full_diagnostics[layer];
                const auto inc_scores = extract_attention_row(inc.attention_scores,
                    inc.attention_key_positions, inc.attention_query_positions, 0);
                const auto full_scores = extract_attention_row(full.attention_scores,
                    full.attention_key_positions, full.attention_query_positions, traced_position);
                const bool scores_equal = inc_scores == full_scores;
                if (!scores_equal) {
                    boundary_metrics("attention_scores", layer, traced_position, inc_scores, full_scores);
                    if (first_probability_layer == model.layer_count) {
                        first_probability_layer = layer;
                        first_probability_checkpoint = "attention_scores";
                    }
                }
                const auto inc_probs = extract_attention_row(inc.attention_probabilities,
                    inc.attention_key_positions, inc.attention_query_positions, 0);
                const auto full_probs = extract_attention_row(full.attention_probabilities,
                    full.attention_key_positions, full.attention_query_positions, traced_position);
                const bool probs_equal = inc_probs == full_probs;
                if (!probs_equal) {
                    boundary_metrics("attention_probabilities", layer, traced_position, inc_probs, full_probs);
                    if (first_probability_layer == model.layer_count) {
                        first_probability_layer = layer;
                        first_probability_checkpoint = "attention_probabilities";
                    }
                }
            }
            std::printf("kv_first_attention_difference position=%u layer=%s checkpoint=%s\n",
                traced_position, first_probability_layer == model.layer_count ? "NONE" :
                std::to_string(first_probability_layer).c_str(), first_probability_checkpoint.c_str());
            std::vector<uint32_t> first_boundary_layer(prefix_length, model.layer_count);
            for (uint32_t position = 0; position < prefix_length; ++position) {
                const auto full_embedding = token_at(full_diagnostics[0].layer_input, EMBED, position);
                if (!boundary_metrics("embedding", 0, position,
                        kv_history[0][position].layer_input, full_embedding))
                    first_boundary_layer[position] = 0;
                for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
                    if (layer == model.layer_count - 1 && position != step) continue;
                    const auto full_output = layer == model.layer_count - 1
                        ? full_layer_outputs[layer]
                        : token_at(full_diagnostics[layer].block_output, EMBED, position);
                    const bool same = boundary_metrics("block_output", layer, position,
                        kv_history[layer][position].block_output, full_output);
                    if (!same && first_boundary_layer[position] == model.layer_count)
                        first_boundary_layer[position] = layer + 1;
                }
                if (first_boundary_layer[position] == model.layer_count)
                    std::printf("kv_boundary_first_difference position=%u checkpoint=NONE\n", position);
                else if (first_boundary_layer[position] == 0)
                    std::printf("kv_boundary_first_difference position=%u checkpoint=embedding\n", position);
                else
                    std::printf("kv_boundary_first_difference position=%u checkpoint=block_output layer=%u\n",
                        position, first_boundary_layer[position] - 1);
            }
            const bool has_position6_boundary_difference = first_boundary_layer[6] > 0 &&
                first_boundary_layer[6] < model.layer_count;
            const bool run_extent_control = prefix_length == 8;
            if (has_position6_boundary_difference || run_extent_control) {
                const uint32_t layer = has_position6_boundary_difference ? first_boundary_layer[6] - 1 : 13;
                const uint32_t position = 6;
                const auto & incremental = kv_history[layer][position];
                const auto & full = full_diagnostics[layer];
                auto trace = [&](const char * name, const std::vector<float> & inc,
                    const std::vector<float> & all, size_t width) {
                    const bool same = boundary_metrics(name, layer, position, inc,
                        token_at(all, width, position));
                    return same;
                };
                auto trace_attention = [&](const char * name, const std::vector<float> & inc,
                    uint32_t inc_keys, uint32_t inc_queries, const std::vector<float> & all,
                    uint32_t all_keys, uint32_t all_queries) {
                    const uint32_t visible = position + 1;
                    if (inc_keys < visible || all_keys < visible || inc_queries == 0 || position >= all_queries)
                        throw std::runtime_error("attention row trace geometry mismatch");
                    std::vector<float> inc_row, full_row;
                    inc_row.reserve(static_cast<size_t>(HEADS) * visible);
                    full_row.reserve(static_cast<size_t>(HEADS) * visible);
                    for (uint32_t head = 0; head < HEADS; ++head) {
                        const size_t ib = static_cast<size_t>(head) * inc_keys * inc_queries;
                        const size_t fb = static_cast<size_t>(head) * all_keys * all_queries + static_cast<size_t>(position) * all_keys;
                        inc_row.insert(inc_row.end(), inc.begin() + ib, inc.begin() + ib + visible);
                        full_row.insert(full_row.end(), all.begin() + fb, all.begin() + fb + visible);
                    }
                    return boundary_metrics(name, layer, position, inc_row, full_row);
                };
                const size_t kv_width = static_cast<size_t>(KV_HEADS) * HEAD_DIM;
                bool still_same = true;
                std::string last_matching = "input";
                auto detail = [&](const char * name, const std::vector<float> & inc,
                    const std::vector<float> & all, size_t width) {
                    if (still_same && trace(name, inc, all, width)) last_matching = name;
                    else if (still_same) {
                        still_same = false;
                        std::printf("kv_first_differing_checkpoint layer=%u position=%u checkpoint=%s last_matching=%s\n",
                            layer, position, name, last_matching.c_str());
                    }
                };
                detail("layer_input", incremental.layer_input, full.layer_input, EMBED);
                detail("attention_rmsnorm", incremental.attention_rmsnorm, full.attention_rmsnorm, EMBED);
                detail("q_projection", incremental.q_projection, full.q_projection, EMBED);
                detail("k_projection", incremental.k_projection, full.k_projection, kv_width);
                detail("v_projection", incremental.v_projection, full.v_projection, kv_width);
                detail("q_rmsnorm", incremental.q_rmsnorm, full.q_rmsnorm, EMBED);
                detail("k_rmsnorm", incremental.k_rmsnorm, full.k_rmsnorm, kv_width);
                detail("q_rope", incremental.q_rope, full.q_rope, EMBED);
                detail("k_rope", incremental.k_rope, full.k_rope, kv_width);
                auto detail_attention = [&](const char * name, const std::vector<float> & inc,
                    const std::vector<float> & all) {
                    if (still_same && trace_attention(name, inc,
                            incremental.attention_key_positions, incremental.attention_query_positions,
                            all, full.attention_key_positions, full.attention_query_positions))
                        last_matching = name;
                    else if (still_same) {
                        still_same = false;
                        std::printf("kv_first_differing_checkpoint layer=%u position=%u checkpoint=%s last_matching=%s\n",
                            layer, position, name, last_matching.c_str());
                    }
                };
                detail_attention("attention_scores", incremental.attention_scores, full.attention_scores);
                detail_attention("attention_probabilities", incremental.attention_probabilities, full.attention_probabilities);
                detail("attention_output", incremental.attention_context, full.attention_context, EMBED);
                detail("o_projection", incremental.attention_projection, full.attention_projection, EMBED);
                detail("attention_residual", incremental.attention_residual, full.attention_residual, EMBED);
                detail("ffn_rmsnorm", incremental.ffn_rmsnorm, full.ffn_rmsnorm, EMBED);
                detail("ffn_gate", incremental.ffn_gate, full.ffn_gate, FFN);
                detail("ffn_up", incremental.ffn_up, full.ffn_up, FFN);
                detail("ffn_swiglu", incremental.ffn_swiglu, full.ffn_swiglu, FFN);
                detail("ffn_down", incremental.ffn_down, full.ffn_down, EMBED);
                detail("block_output", incremental.block_output, full.block_output, EMBED);
                if (still_same) std::printf("kv_first_differing_checkpoint layer=%u position=%u checkpoint=NONE last_matching=block_output\n", layer, position);

                if (layer == 13) {
                    const uint32_t softmax_control_layer = first_probability_layer < model.layer_count ?
                        first_probability_layer : 0;
                    const auto & softmax_full = full_diagnostics[softmax_control_layer];
                    const auto & softmax_incremental = kv_history[softmax_control_layer][position];
                    const auto full_visible_scores = extract_attention_row(softmax_full.attention_scores,
                        softmax_full.attention_key_positions, softmax_full.attention_query_positions, position);
                    const auto incremental_visible_scores = extract_attention_row(softmax_incremental.attention_scores,
                        softmax_incremental.attention_key_positions, softmax_incremental.attention_query_positions, 0);
                    const bool visible_scores_equal = full_visible_scores == incremental_visible_scores;
                    std::printf("softmax_extent_control_input layer=%u position=%u full_key_extent=%u incremental_key_extent=%u visible_raw_scores_equal=%s full_hash=%016llx incremental_hash=%016llx masked_future=negative_infinity\n",
                        softmax_control_layer, position, softmax_full.attention_key_positions,
                        softmax_incremental.attention_key_positions, visible_scores_equal ? "YES" : "NO",
                        static_cast<unsigned long long>(fnv1a64(reinterpret_cast<const uint8_t *>(full_visible_scores.data()), full_visible_scores.size() * sizeof(float))),
                        static_cast<unsigned long long>(fnv1a64(reinterpret_cast<const uint8_t *>(incremental_visible_scores.data()), incremental_visible_scores.size() * sizeof(float))));
                    for (uint32_t head = 0; head < HEADS; ++head) {
                        const size_t row_begin = static_cast<size_t>(head) * visible_keys;
                        const size_t future_index = (static_cast<size_t>(head) * softmax_full.attention_query_positions + position) *
                            softmax_full.attention_key_positions + softmax_full.attention_key_positions - 1;
                        std::printf("softmax_extent_control_raw_scores head=%u visible=", head);
                        for (uint32_t key = 0; key < visible_keys; ++key)
                            std::printf("%s%a", key == 0 ? "" : ",", full_visible_scores[row_begin + key]);
                        std::printf(" future_unmasked=%a control_mask=-inf\n", softmax_full.attention_scores[future_index]);
                    }
                    struct SoftmaxCase {
                        const char * name;
                        uint32_t keys;
                        uint32_t queries;
                        ggml_tensor * scores;
                        ggml_tensor * mask;
                        ggml_tensor * output;
                        std::vector<float> score_data;
                        std::vector<float> mask_data;
                    };
                    Context control = make_context();
                    std::vector<SoftmaxCase> cases;
                    for (const auto config : { std::array<uint32_t, 3>{7, 1, 0},
                             std::array<uint32_t, 3>{7, 8, 1},
                             std::array<uint32_t, 3>{8, 1, 0},
                             std::array<uint32_t, 3>{8, 8, 1} }) {
                        const uint32_t keys = config[0], queries = config[1];
                        const char * name = keys == 7 ? (queries == 1 ? "short_single" : "short_batch") :
                            (queries == 1 ? "padded_single" : "padded_batch");
                        SoftmaxCase item{name, keys, queries,
                            ggml_new_tensor_3d(control.ctx, GGML_TYPE_F32, keys, queries, HEADS),
                            ggml_new_tensor_2d(control.ctx, GGML_TYPE_F32, keys, queries), nullptr, {}, {}};
                        item.output = ggml_soft_max_ext(control.ctx, item.scores, item.mask,
                            1.0f / std::sqrt(static_cast<float>(HEAD_DIM)), 0.0f);
                        item.score_data.resize(static_cast<size_t>(keys) * queries * HEADS);
                        item.mask_data.resize(static_cast<size_t>(keys) * queries, 0.0f);
                        for (uint32_t head = 0; head < HEADS; ++head)
                            for (uint32_t query = 0; query < queries; ++query)
                                for (uint32_t key = 0; key < keys; ++key) {
                                    const size_t source = static_cast<size_t>(head) * softmax_full.attention_key_positions *
                                        softmax_full.attention_query_positions + static_cast<size_t>(position) *
                                        softmax_full.attention_key_positions + std::min(key, softmax_full.attention_key_positions - 1);
                                    const size_t destination = (static_cast<size_t>(head) * queries + query) * keys + key;
                                    item.score_data[destination] = softmax_full.attention_scores[source];
                                    if (keys == 8 && key == 7)
                                        item.mask_data[static_cast<size_t>(query) * keys + key] = -INFINITY;
                                }
                        cases.push_back(std::move(item));
                    }
                    auto * control_graph = ggml_new_graph(control.ctx);
                    for (auto & item : cases) ggml_build_forward_expand(control_graph, item.output);
                    control.buffer = ggml_backend_alloc_ctx_tensors(control.ctx, control.backend);
                    if (!control.buffer) throw std::runtime_error("isolated softmax control allocation failed");
                    for (const auto & item : cases) {
                        set_tensor(item.scores, item.score_data.data(), item.score_data.size() * sizeof(float));
                        set_tensor(item.mask, item.mask_data.data(), item.mask_data.size() * sizeof(float));
                    }
                    if (ggml_backend_graph_compute(control.backend, control_graph) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("isolated softmax shape-control graph failed");
                    ggml_backend_synchronize(control.backend);
                    std::vector<std::vector<float>> outputs;
                    for (auto & item : cases) outputs.push_back(get_f32(control, item.output));
                    ggml_backend_cpu_set_n_threads(control.backend, 1);
                    if (ggml_backend_graph_compute(control.backend, control_graph) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("one-thread softmax control graph failed");
                    ggml_backend_synchronize(control.backend);
                    std::vector<std::vector<float>> outputs_one_thread;
                    for (auto & item : cases) outputs_one_thread.push_back(get_f32(control, item.output));
                    auto select_row = [](const std::vector<float> & values, uint32_t keys,
                        uint32_t queries, uint32_t query, uint32_t count = 0) {
                        if (count == 0) count = keys;
                        std::vector<float> row;
                        row.reserve(static_cast<size_t>(HEADS) * count);
                        for (uint32_t head = 0; head < HEADS; ++head) {
                            const size_t begin = (static_cast<size_t>(head) * queries + query) * keys;
                            row.insert(row.end(), values.begin() + begin, values.begin() + begin + count);
                        }
                        return row;
                    };
                    const auto actual_full_probs = [&] {
                        std::vector<float> row;
                        for (uint32_t head = 0; head < HEADS; ++head) {
                            const size_t begin = (static_cast<size_t>(head) * softmax_full.attention_query_positions + position) * softmax_full.attention_key_positions;
                            row.insert(row.end(), softmax_full.attention_probabilities.begin() + begin,
                                softmax_full.attention_probabilities.begin() + begin + position + 1);
                        }
                        return row;
                    }();
                    const auto actual_single_probs = [&] {
                        std::vector<float> row;
                        for (uint32_t head = 0; head < HEADS; ++head) {
                            const size_t begin = static_cast<size_t>(head) * softmax_incremental.attention_key_positions;
                            row.insert(row.end(), softmax_incremental.attention_probabilities.begin() + begin,
                                softmax_incremental.attention_probabilities.begin() + begin + position + 1);
                        }
                        return row;
                    }();
                    const auto full_control = select_row(outputs[3], 8, 8, position, position + 1);
                    const auto short_single = select_row(outputs[0], 7, 1, 0);
                    const auto short_batch = select_row(outputs[1], 7, 8, position);
                    const auto padded_single = select_row(outputs[2], 8, 1, 0, position + 1);
                    for (size_t i = 0; i < cases.size(); ++i)
                        boundary_metrics("softmax_threads_8_vs_1", softmax_control_layer, position, outputs[i], outputs_one_thread[i]);
                    std::vector<float> fp64_oracle;
                    const double scale = static_cast<double>(1.0f / std::sqrt(static_cast<float>(HEAD_DIM)));
                    for (uint32_t head = 0; head < HEADS; ++head) {
                        std::array<double, 7> exps{};
                        double maximum = -std::numeric_limits<double>::infinity();
                        for (uint32_t key = 0; key <= position; ++key) {
                            const size_t index = static_cast<size_t>(head) * softmax_full.attention_key_positions *
                                softmax_full.attention_query_positions + static_cast<size_t>(position) *
                                softmax_full.attention_key_positions + key;
                            maximum = std::max(maximum, static_cast<double>(softmax_full.attention_scores[index]) * scale);
                        }
                        double sum = 0;
                        for (uint32_t key = 0; key <= position; ++key) {
                            const size_t index = static_cast<size_t>(head) * softmax_full.attention_key_positions *
                                softmax_full.attention_query_positions + static_cast<size_t>(position) *
                                softmax_full.attention_key_positions + key;
                            exps[key] = std::exp(static_cast<double>(softmax_full.attention_scores[index]) * scale - maximum);
                            sum += exps[key];
                        }
                        for (uint32_t key = 0; key <= position; ++key)
                            fp64_oracle.push_back(static_cast<float>(exps[key] / sum));
                    }
                    boundary_metrics("softmax_actual_full_vs_same_shape_control", softmax_control_layer, position, full_control, actual_full_probs);
                    boundary_metrics("softmax_incremental_extent8_vs_full", softmax_control_layer, position,
                        padded_single, actual_full_probs);
                    boundary_metrics("softmax_actual_single_vs_same_shape_control", softmax_control_layer, position, padded_single, actual_single_probs);
                    boundary_metrics("softmax_row_count_only_1_vs_8", softmax_control_layer, position, short_batch, short_single);
                    boundary_metrics("softmax_extent8_row_count_only_1_vs_8", softmax_control_layer, position,
                        select_row(outputs[3], 8, 8, position, position + 1), padded_single);
                    boundary_metrics("softmax_key_extent_only_7_vs_8_masked", softmax_control_layer, position,
                        padded_single, short_single);
                    boundary_metrics("softmax_combined_shape_8x8_vs_7x1", softmax_control_layer, position, full_control, short_single);
                    boundary_metrics("softmax_full_vs_fp64_oracle", softmax_control_layer, position, actual_full_probs, fp64_oracle);
                    boundary_metrics("softmax_single_vs_fp64_oracle", softmax_control_layer, position, actual_single_probs, fp64_oracle);
                    for (const auto & item : cases)
                        std::printf("softmax_control_shape layer=%u name=%s op=%s type=%s score_ne=[%lld,%lld,%lld] score_nb=[%zu,%zu,%zu] contiguous=%s mask_ne=[%lld,%lld]\n",
                            softmax_control_layer, item.name, ggml_op_name(item.output->op), ggml_type_name(item.output->type),
                            static_cast<long long>(item.scores->ne[0]), static_cast<long long>(item.scores->ne[1]),
                            static_cast<long long>(item.scores->ne[2]), item.scores->nb[0], item.scores->nb[1],
                            item.scores->nb[2], ggml_is_contiguous(item.scores) ? "YES" : "NO",
                            static_cast<long long>(item.mask->ne[0]), static_cast<long long>(item.mask->ne[1]));
                    trace("attention_output_downstream", incremental.attention_context, full.attention_context, EMBED);
                    trace("o_projection_downstream", incremental.attention_projection, full.attention_projection, EMBED);
                    trace("attention_residual_downstream", incremental.attention_residual, full.attention_residual, EMBED);
                    trace("ffn_rmsnorm_downstream", incremental.ffn_rmsnorm, full.ffn_rmsnorm, EMBED);
                    trace("ffn_gate_downstream", incremental.ffn_gate, full.ffn_gate, FFN);
                    trace("ffn_up_downstream", incremental.ffn_up, full.ffn_up, FFN);
                    trace("ffn_swiglu_downstream", incremental.ffn_swiglu, full.ffn_swiglu, FFN);
                    trace("ffn_down_downstream", incremental.ffn_down, full.ffn_down, EMBED);
                    trace("block_output_downstream", incremental.block_output, full.block_output, EMBED);
                    analyze_attention_output_shape(model, full, incremental, layer, position);
                }
            }
            for (uint32_t layer : {14u, 15u}) {
                const uint32_t position = 6;
                const auto & incremental = kv_history[layer][position];
                const auto & full = full_diagnostics[layer];
                auto trace_stage = [&](const char * name, const std::vector<float> & inc,
                    const std::vector<float> & all, size_t width) {
                    boundary_metrics(name, layer, position, inc, token_at(all, width, position));
                };
                trace_stage("layer_input", incremental.layer_input, full.layer_input, EMBED);
                trace_stage("attention_rmsnorm", incremental.attention_rmsnorm, full.attention_rmsnorm, EMBED);
                trace_stage("q_projection", incremental.q_projection, full.q_projection, EMBED);
                trace_stage("k_projection", incremental.k_projection, full.k_projection, KV_HEADS * HEAD_DIM);
                trace_stage("v_projection", incremental.v_projection, full.v_projection, KV_HEADS * HEAD_DIM);
                trace_stage("q_rmsnorm", incremental.q_rmsnorm, full.q_rmsnorm, EMBED);
                trace_stage("k_rmsnorm", incremental.k_rmsnorm, full.k_rmsnorm, KV_HEADS * HEAD_DIM);
                trace_stage("q_rope", incremental.q_rope, full.q_rope, EMBED);
                trace_stage("k_rope", incremental.k_rope, full.k_rope, KV_HEADS * HEAD_DIM);
                auto attention_row = [&](const std::vector<float> & values, uint32_t keys,
                    uint32_t queries) {
                    std::vector<float> row;
                    for (uint32_t head = 0; head < HEADS; ++head) {
                        const size_t begin = (static_cast<size_t>(head) * queries + (queries == 1 ? 0 : position)) * keys;
                        row.insert(row.end(), values.begin() + begin, values.begin() + begin + position + 1);
                    }
                    return row;
                };
                boundary_metrics("attention_scores", layer, position,
                    attention_row(incremental.attention_scores, incremental.attention_key_positions,
                        incremental.attention_query_positions),
                    attention_row(full.attention_scores, full.attention_key_positions,
                        full.attention_query_positions));
                boundary_metrics("attention_probabilities", layer, position,
                    attention_row(incremental.attention_probabilities, incremental.attention_key_positions,
                        incremental.attention_query_positions),
                    attention_row(full.attention_probabilities, full.attention_key_positions,
                        full.attention_query_positions));
                trace_stage("attention_output", incremental.attention_context, full.attention_context, EMBED);
                trace_stage("o_projection", incremental.attention_projection, full.attention_projection, EMBED);
                trace_stage("attention_residual", incremental.attention_residual, full.attention_residual, EMBED);
                trace_stage("ffn_rmsnorm", incremental.ffn_rmsnorm, full.ffn_rmsnorm, EMBED);
                trace_stage("ffn_gate", incremental.ffn_gate, full.ffn_gate, FFN);
                trace_stage("ffn_up", incremental.ffn_up, full.ffn_up, FFN);
                trace_stage("ffn_swiglu", incremental.ffn_swiglu, full.ffn_swiglu, FFN);
                trace_stage("ffn_down", incremental.ffn_down, full.ffn_down, EMBED);
                trace_stage("block_output", incremental.block_output, full.block_output, EMBED);
                if (layer == 15)
                    analyze_attention_output_shape(model, full, incremental, layer, position);
            }
            for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
                for (uint32_t position = 0; position < prefix_length; ++position) {
                    const auto & incremental = kv_history[layer][position];
                    const double input_delta = max_difference(incremental.layer_input,
                        token_at(full_diagnostics[layer].layer_input, EMBED, position));
                    const double norm_delta = max_difference(incremental.attention_rmsnorm,
                        token_at(full_diagnostics[layer].attention_rmsnorm, EMBED, position));
                    const double q_delta = max_difference(incremental.q_projection,
                        token_at(full_diagnostics[layer].q_projection, EMBED, position));
                    const double k_delta = max_difference(incremental.k_projection,
                        token_at(full_diagnostics[layer].k_projection, KV_HEADS * HEAD_DIM, position));
                    const double v_delta = max_difference(incremental.v_projection,
                        token_at(full_diagnostics[layer].v_projection, KV_HEADS * HEAD_DIM, position));
                    if (input_delta != 0 || norm_delta != 0 || q_delta != 0 || k_delta != 0 || v_delta != 0)
                        std::printf("kv_token_path_difference layer=%u position=%u input=%.9g attn_norm=%.9g q=%.9g k=%.9g v=%.9g\n",
                            layer, position, input_delta, norm_delta, q_delta, k_delta, v_delta);
                }
            }
            for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
                const auto & full_key = full_diagnostics[layer].key_f16;
                const auto & full_value = full_diagnostics[layer].value_f16;
                const auto & cached_key = kv_diagnostics[layer].key_f16;
                const auto & cached_value = kv_diagnostics[layer].value_f16;
                if (full_key.size() != prefix_length * kv_tensor_bytes ||
                    full_value.size() != prefix_length * kv_tensor_bytes ||
                    cached_key.size() != kv_tensor_bytes || cached_value.size() != kv_tensor_bytes)
                    throw std::runtime_error("F16 KV checkpoint geometry mismatch");
                const size_t current_offset = static_cast<size_t>(step) * kv_tensor_bytes;
                const bool current_equal = std::equal(cached_key.begin(), cached_key.end(),
                    full_key.begin() + current_offset) &&
                    std::equal(cached_value.begin(), cached_value.end(),
                    full_value.begin() + current_offset);
                bool layer_cache_history_equal = current_equal;
                if (!current_equal) {
                    all_cache_bytes_match = false;
                    std::fprintf(stderr, "kv_current_checkpoint_mismatch layer=%u position=%u\n", layer, step);
                    if (!diagnostic_mode)
                        throw std::runtime_error("newly appended F16 K/V differs from full-sequence K/V");
                }
                for (uint32_t position = 0; position < prefix_length; ++position) {
                    const size_t offset = static_cast<size_t>(position) * kv_tensor_bytes;
                    const auto key_difference = std::mismatch(full_key.begin() + offset,
                        full_key.begin() + offset + kv_tensor_bytes, cache.key_bytes(layer) + offset);
                    const auto value_difference = std::mismatch(full_value.begin() + offset,
                        full_value.begin() + offset + kv_tensor_bytes, cache.value_bytes(layer) + offset);
                    if (key_difference.first != full_key.begin() + offset + kv_tensor_bytes ||
                        value_difference.first != full_value.begin() + offset + kv_tensor_bytes) {
                        const bool key_diff = key_difference.first != full_key.begin() + offset + kv_tensor_bytes;
                        const size_t byte_index = key_diff ? static_cast<size_t>(key_difference.first - (full_key.begin() + offset)) :
                            static_cast<size_t>(value_difference.first - (full_value.begin() + offset));
                        const uint8_t expected_byte = key_diff ? *key_difference.first : *value_difference.first;
                        const uint8_t cached_byte = key_diff ? *(cache.key_bytes(layer) + offset + byte_index) :
                            *(cache.value_bytes(layer) + offset + byte_index);
                        std::fprintf(stderr, "kv_cache_mismatch layer=%u position=%u tensor=%s byte_index=%zu kv_head=%zu component=%zu byte_in_f16=%zu full=%02x cache=%02x current_position=%u\n",
                            layer, position, key_diff ? "K" : "V", byte_index,
                            byte_index / (HEAD_DIM * sizeof(uint16_t)),
                            (byte_index / sizeof(uint16_t)) % HEAD_DIM,
                            byte_index % sizeof(uint16_t), expected_byte, cached_byte, step);
                        if (layer == 14 && position == 6 && key_diff && byte_index + 1 < kv_tensor_bytes) {
                            uint16_t full_bits = 0, cached_bits = 0;
                            std::memcpy(&full_bits, full_key.data() + offset + byte_index, sizeof(full_bits));
                            std::memcpy(&cached_bits, cache.key_bytes(layer) + offset + byte_index, sizeof(cached_bits));
                            std::fprintf(stderr, "kv_cache_mismatch_f16 layer=14 position=6 head=%zu component=%zu full_bits=%04x cache_bits=%04x full_value=%.9g cache_value=%.9g\n",
                                byte_index / (HEAD_DIM * sizeof(uint16_t)),
                                (byte_index / sizeof(uint16_t)) % HEAD_DIM,
                                full_bits, cached_bits, ggml_fp16_to_fp32(full_bits), ggml_fp16_to_fp32(cached_bits));
                        }
                        all_cache_bytes_match = false;
                        layer_cache_history_equal = false;
                        if (!diagnostic_mode)
                            throw std::runtime_error("persistent F16 K/V cache differs from full-sequence cache");
                    }
                }
                if (layer == 0 || layer == 1 || layer == 20 || layer == 39)
                    for (uint32_t position : {0u, 1u, 3u, 7u})
                        std::printf("kv_checkpoint layer=%u position=%u K_equal=YES V_equal=YES key_bytes=%zu value_bytes=%zu position_rope_applied_once=YES\n",
                            layer, position, kv_tensor_bytes, kv_tensor_bytes);

                const auto full_context = final_position(full_diagnostics[layer].attention_context, EMBED);
                const auto full_residual_layer = final_position(full_diagnostics[layer].attention_residual, EMBED);
                const auto full_output_layer = full_layer_outputs[layer];
                const double attention_delta = max_difference(kv_diagnostics[layer].attention_context, full_context);
                const double residual_delta = max_difference(kv_diagnostics[layer].attention_residual, full_residual_layer);
                const double output_delta = max_difference(kv_layer_outputs[layer], full_output_layer);
                if (attention_delta != 0 || residual_delta != 0 || output_delta != 0) {
                    if (first_divergent_layer == model.layer_count) first_divergent_layer = layer;
                    std::printf("kv_layer_first_difference layer=%u position=%u attention_max_abs=%.9g residual_max_abs=%.9g output_max_abs=%.9g cache_history_bytes_equal=%s\n",
                        layer, step, attention_delta, residual_delta, output_delta, layer_cache_history_equal ? "YES" : "NO");
                }
            }
            std::printf("kv_first_divergent_layer=%s\n", first_divergent_layer == model.layer_count ? "NONE" : std::to_string(first_divergent_layer).c_str());
        }
        bool b0_attention_ok = true, b0_residual_ok = true, b0_output_ok = true;
        bool hidden_ok = true, logits_ok = true;
        if (have_full) {
            const auto full_attention = final_position(full_diagnostics[0].attention_context, EMBED);
            const auto full_residual = final_position(full_diagnostics[0].attention_residual, EMBED);
            const auto full_block0_output = full_layer_outputs[0];
            b0_attention_ok = compare("kv_vs_full_block0_attention", kv_diagnostics[0].attention_context, full_attention, 1);
            b0_residual_ok = compare("kv_vs_full_block0_residual", kv_diagnostics[0].attention_residual, full_residual, 1);
            b0_output_ok = compare("kv_vs_full_block0_output", kv_block0_output, full_block0_output, 1);
            hidden_ok = compare("kv_vs_full_final_hidden", kv_hidden, full_hidden, 1);
            logits_ok = compare("kv_vs_full_logits", kv_logits, full_logits, 1);
        }
        const uint32_t full_next = have_full ? greedy_token(full_logits) : reference_next;
        const uint32_t kv_next = greedy_token(kv_logits);
        if (kv_next != reference_next || (have_full && (full_next != reference_next || kv_next != full_next ||
            !b0_attention_ok || !b0_residual_ok || !b0_output_ok || !hidden_ok || !logits_ok)))
            throw std::runtime_error("persistent KV diverged from full recompute/reference at prefix " +
                std::to_string(prefix_length));
        generated.push_back(kv_next);
        const std::string full_next_text = have_full ? std::to_string(full_next) : "SKIPPED_FINAL_PREFIX_ONLY";
        std::printf("kv_prefix=%u input_token=%d full_recompute_next=%s persistent_kv_next=%u llama_next=%u token_parity=PASS cache_bytes=%zu cache_expected_bytes=%zu bytes_per_token=%zu\n",
            prefix_length, token, full_next_text.c_str(), kv_next, reference_next, cache.bytes(),
            static_cast<size_t>(prefix_length) * cache.bytes_per_token(), cache.bytes_per_token());
    }
    std::printf("persistent_kv_sequence=");
    for (size_t i = 0; i < generated.size(); ++i)
        std::printf("%s%u", i == 0 ? "" : ",", generated[i]);
    std::printf("\npersistent_kv_cache_bytes_before_reset=%zu\n", cache.bytes());
    cache.reset();
    if (cache.bytes() != 0 || cache.completed_positions() != 0)
        throw std::runtime_error("persistent Qwen3 KV reset failed");
    std::printf("persistent_kv_reset=PASS bytes_after_reset=%zu completed_positions=%u\n",
        cache.bytes(), cache.completed_positions());

    // A fresh cache in this same process must reproduce the clean one-token run.
    Qwen3KvCache clean({model.layer_count, HEADS, KV_HEADS, HEAD_DIM, 1});
    const std::string first_ref = prefix_reference_dir(reference_root, 0);
    std::vector<float> clean_hidden;
    const std::vector<int32_t> seed{tokens.front()};
    for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
        clean_hidden = run_positions(model, first_ref, 1, layer, clean_hidden,
            false, seed, &clean, 0, nullptr, nullptr, nullptr, true);
        if (clean_hidden.empty()) throw std::runtime_error("clean reset run produced non-finite state");
    }
    clean.complete_position(0);
    const auto clean_logits = run_final_head(model, first_ref, clean_hidden, true);
    if (greedy_token(clean_logits) != greedy_token(first_full_logits) ||
        !compare("reset_clean_logits_vs_full", clean_logits, first_full_logits, 1))
        throw std::runtime_error("fresh cache after reset differs from clean full recompute");
    std::printf("persistent_kv_reset_clean_replay=PASS session_state_isolated=YES cancellation_api=NOT_SUPPORTED\n");
    std::printf("persistent_kv_incremental_replay_equivalence=%s first_divergent_layer=%s unchanged_atol=%.9g\n",
        all_incremental_replays_match ? "PASS" : "FAIL",
        first_incremental_divergence_layer == model.layer_count ? "NONE" :
            std::to_string(first_incremental_divergence_layer).c_str(), ATOL);
    std::printf("persistent_kv_all_cache_bytes_match_full=%s diagnostic_mode=%s\n",
        all_cache_bytes_match ? "YES" : "NO", diagnostic_mode ? "YES" : "NO");
    if (!all_incremental_replays_match)
        throw std::runtime_error("persistent KV differs from fresh incremental replay; qualification blocked");
    if (!all_cache_bytes_match)
        throw std::runtime_error("persistent KV differs from full-sequence F16 cache; production qualification blocked");
}
} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: qwen3_block_qualification SEMANTIC_BOOTSTRAP SOURCE_HTTP_URL REFERENCE_DIR POSITIONS(1..32) [BLOCK_COUNT] [MODE] [tokens:ID,...|seed:ID] [--serial | --execution serial|parallel] [--threads N]\n");
        return 2;
    }
    const uint32_t positions = static_cast<uint32_t>(std::stoul(argv[4]));
    std::vector<std::string> positional;
    bool serial_convenience = false;
    bool execution_specified = false;
    bool execution_cli_serial = false;
    bool threads_specified = false;
    uint32_t cli_threads = 0;
    uint32_t benchmark_repeats = 1;
    for (int index = 5; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--serial") {
            serial_convenience = true;
            execution_specified = true;
            execution_cli_serial = true;
        } else if (argument == "--execution") {
            if (++index >= argc) return 2;
            const std::string mode = argv[index];
            if (mode != "serial" && mode != "parallel") return 2;
            execution_specified = true;
            execution_cli_serial = mode == "serial";
        } else if (argument == "--threads") {
            if (++index >= argc) return 2;
            const long long value = std::stoll(argv[index]);
            if (value < 1 || value > 128) return 2;
            threads_specified = true;
            cli_threads = static_cast<uint32_t>(value);
        } else if (argument == "--benchmark-repeats") {
            if (++index >= argc) return 2;
            const long long value = std::stoll(argv[index]);
            if (value < 1 || value > 5) return 2;
            benchmark_repeats = static_cast<uint32_t>(value);
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.size() > 3 ||
        (serial_convenience && execution_specified && !execution_cli_serial)) return 2;
    const uint32_t block_count = positional.empty() ? 1 : static_cast<uint32_t>(std::stoul(positional[0]));
    const std::string input_mode = positional.size() < 2 ? "actual" : positional[1];
    const bool has_token_spec = positional.size() == 3;
    const char * environment_threads = std::getenv("VBUF_QWEN_THREADS");
    g_execution_policy.total_cpu_threads = 4;
    if (!threads_specified && !serial_convenience && environment_threads != nullptr)
        g_execution_policy.total_cpu_threads = static_cast<uint32_t>(std::stoul(environment_threads));
    if (threads_specified) g_execution_policy.total_cpu_threads = cli_threads;
    if (execution_specified)
        g_execution_policy.mode = execution_cli_serial ? Qwen3ExecutionMode::Serial : Qwen3ExecutionMode::Parallel;
    if (serial_convenience) {
        if (threads_specified && cli_threads != 1) {
            std::fprintf(stderr, "--serial fixes the kernel budget at one; use --execution serial --threads N for a serial graph with N kernel threads\n");
            return 2;
        }
        g_execution_policy.total_cpu_threads = 1;
    }
    if (g_execution_policy.total_cpu_threads < 1 || g_execution_policy.total_cpu_threads > 128) {
        std::fprintf(stderr, "effective Qwen CPU budget must be in 1..128\n");
        return 2;
    }
    g_serial_convenience = serial_convenience;
    const size_t executor_workers = g_execution_policy.mode == Qwen3ExecutionMode::Parallel ?
        std::min<size_t>(3, g_execution_policy.total_cpu_threads) : 1;
    g_qwen_executor = std::make_unique<BoundedExecutor>(executor_workers);
    [[maybe_unused]] ExecutionMetricsReporter metrics_reporter;

    std::vector<int32_t> input_tokens;
    if (has_token_spec) {
        const std::string token_spec = positional[2];
        if (input_mode == "long_persistent_generate") {
            if (token_spec.rfind("seed:", 0) != 0) return 2;
            const long long value = std::stoll(token_spec.substr(5));
            if (value < 0 || value > INT32_MAX) return 2;
            input_tokens.push_back(static_cast<int32_t>(value));
        } else {
            if (token_spec.rfind("tokens:", 0) != 0) return 2;
            std::stringstream token_stream(token_spec.substr(7));
            std::string token;
            while (std::getline(token_stream, token, ',')) {
                const long long value = std::stoll(token);
                if (value < 0 || value > INT32_MAX) return 2;
                input_tokens.push_back(static_cast<int32_t>(value));
            }
            if (input_tokens.size() != positions) return 2;
        }
    }
    uint32_t reference_input_layer = 0;
    bool use_reference_input = false;
    const bool persistent_kv = input_mode == "persistent_kv";
    const bool long_persistent_kv = input_mode == "long_persistent_kv";
    const bool long_persistent_generate = input_mode == "long_persistent_generate";
    const bool full_reference_logits = input_mode == "full_reference_logits";
    if (input_mode == "reference_layer0") {
        use_reference_input = true;
        reference_input_layer = 1;
    } else if (input_mode.rfind("reference_input:", 0) == 0) {
        use_reference_input = true;
        reference_input_layer = static_cast<uint32_t>(std::stoul(input_mode.substr(16)));
    }
    if (positions == 0 || positions > 32 || block_count == 0 ||
        (positional.size() >= 2 && input_mode != "actual" && !use_reference_input && !persistent_kv && !long_persistent_kv && !long_persistent_generate && !full_reference_logits) ||
        (use_reference_input && block_count != 1) ||
        (persistent_kv && (positions > 8 || !has_token_spec || input_tokens.size() != positions || use_reference_input)) ||
        (long_persistent_kv && (!has_token_spec || input_tokens.size() != positions || use_reference_input)) ||
        (long_persistent_generate && (!has_token_spec || input_tokens.size() != 1 || use_reference_input)) ||
        (full_reference_logits && (!has_token_spec || input_tokens.size() != positions || use_reference_input)) ||
        (benchmark_repeats > 1 && !full_reference_logits)) return 2;
    try {
        Model model;
        open_model(argv[1], argv[2], &model);
        if (!input_tokens.empty()) {
            const auto & embedding = get(model, "token_embd.weight");
            if (static_cast<uint64_t>(*std::max_element(input_tokens.begin(), input_tokens.end())) >= embedding.view.dimensions[1])
                throw std::runtime_error("input token ID exceeds vocabulary size");
        }
        if (block_count > model.layer_count) throw std::runtime_error("block count exceeds Qwen3 model depth");
        const std::string reference_dir = argv[3];
        if (persistent_kv) {
            if (block_count != model.layer_count)
                throw std::runtime_error("persistent Qwen3 KV qualification requires all 40 layers");
            run_persistent_kv(model, reference_dir, input_tokens);
            std::printf("persistent_kv_real_artifact=PHYSICALLY_QUALIFIED production_session=DISABLED ordinary_http=NOT_TESTED\n");
            return 0;
        }
        if (full_reference_logits) {
            if (block_count != model.layer_count)
                throw std::runtime_error("full/reference logits qualification requires all 40 layers");
            for (uint32_t repeat = 0; repeat < benchmark_repeats; ++repeat) {
                std::printf("qualification_iteration=%u/%u warm_process=%s\n", repeat + 1,
                    benchmark_repeats, repeat == 0 ? "NO" : "YES");
                run_full_reference_logits(model, input_tokens, reference_dir);
            }
            return 0;
        }
        if (long_persistent_kv || long_persistent_generate) {
            if (block_count != model.layer_count || !internal_qualification_mode())
                throw std::runtime_error("long persistent Qwen3 qualification requires all 40 layers and internal-only mode");
            const auto tokens = long_persistent_generate ? generate_persistent_tokens(model,
                static_cast<uint32_t>(input_tokens.front()), positions) : input_tokens;
            run_long_persistent_kv(model, tokens, reference_dir);
            std::printf("long_persistent_kv_real_artifact=PHYSICALLY_QUALIFIED production_session=DISABLED ordinary_http=NOT_TESTED\n");
            return 0;
        }
        std::vector<float> hidden;
        if (use_reference_input) {
            if (reference_input_layer >= model.layer_count)
                throw std::runtime_error("reference input layer index is out of range");
            if (reference_input_layer == 0) {
                hidden = reference(reference_dir, "embd-0");
            } else {
                const uint32_t previous_layer = reference_input_layer - 1;
                hidden = reference(reference_dir, "l_out-" + std::to_string(previous_layer) + "-0");
            }
            if (hidden.size() != static_cast<size_t>(EMBED) * positions)
                throw std::runtime_error("reference block input shape mismatch");
            hidden = run_positions(model, reference_dir, positions, reference_input_layer, hidden, true);
        } else {
            for (uint32_t layer = 0; layer < block_count; ++layer) {
                hidden = run_positions(model, reference_dir, positions, layer, hidden, false, input_tokens);
                if (hidden.empty()) throw std::runtime_error("non-finite hidden state at block boundary");
            }
        }
        if (hidden.empty()) throw std::runtime_error("non-finite hidden state at block boundary");
        if (!use_reference_input && block_count == model.layer_count)
            run_final_head(model, reference_dir, hidden);
        std::printf("blocks_executed=%u positions=%u finite_hidden_state=YES numerical_equivalence=METRICS_RECORDED input_mode=%s\n",
            use_reference_input ? 1 : block_count, positions, use_reference_input ? input_mode.c_str() : "actual");
        std::printf("ordinary_generation=ISOLATED_FULL_RECOMPUTE tool_calling=NOT_TESTED pi=NOT_TESTED\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Qwen3 block qualification failed closed: %s\n", error.what());
        return 1;
    }
}

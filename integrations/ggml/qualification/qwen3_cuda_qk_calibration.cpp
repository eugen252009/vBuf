#include "vbuf_high_precision_reference.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml::reference;
using namespace vbuf_ml::numerics;
namespace fs = std::filesystem;

namespace {
constexpr size_t head_dimension = 128;
constexpr size_t query_heads = 40;
constexpr size_t kv_heads = 8;
constexpr const char * model_identity =
    "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";
constexpr const char * placement = "multi:0x26,1x14;emb=0;norm=1;head=1";
constexpr const char * ggml_commit = "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::map<std::string, std::string> read_metadata(const fs::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open capture metadata: " + path.string());
    std::map<std::string, std::string> result;
    std::string line;
    while (std::getline(input, line)) {
        const size_t equal = line.find('=');
        if (equal != std::string::npos) result[line.substr(0, equal)] = line.substr(equal + 1);
    }
    return result;
}

size_t metadata_size(const std::map<std::string, std::string> & metadata, const std::string & key) {
    const auto found = metadata.find(key);
    if (found == metadata.end()) throw std::runtime_error("capture metadata is missing " + key);
    size_t consumed = 0;
    const size_t value = std::stoull(found->second, &consumed);
    if (consumed != found->second.size()) throw std::runtime_error("invalid capture metadata " + key);
    return value;
}

template<typename T>
std::vector<T> read_elements(std::ifstream & input, size_t count, const std::string & section) {
    std::vector<T> values(count);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) throw std::runtime_error("truncated capture section: " + section);
    return values;
}

void skip_bytes(std::ifstream & input, size_t count, const std::string & section) {
    input.seekg(static_cast<std::streamoff>(count), std::ios::cur);
    if (!input) throw std::runtime_error("truncated capture section: " + section);
}

struct Capture {
    fs::path path;
    std::string phase;
    size_t source_capacity = 0;
    size_t rows = 0;
    size_t visible = 0;
    std::vector<uint16_t> keys_f16;
    std::vector<float> query_f32;
};

Capture read_capture(const fs::path & directory, size_t source_capacity, const std::string & phase) {
    std::ostringstream stem;
    stem << "boundary-capacity-" << source_capacity << '-' << phase << "-layer-00";
    const auto metadata = read_metadata(directory / (stem.str() + ".meta"));
    require(metadata.at("format") == "qwen3-native-av-boundary-v1", "unsupported Qwen boundary capture format");
    require(metadata.at("phase") == phase && metadata_size(metadata, "capacity") == source_capacity &&
        metadata_size(metadata, "layer") == 0 && metadata_size(metadata, "device_id") == 0 &&
        metadata_size(metadata, "query_heads") == query_heads && metadata_size(metadata, "kv_heads") == kv_heads &&
        metadata_size(metadata, "head_dim") == head_dimension && metadata.at("qk_capture") == "yes",
        "capture metadata does not match the layer-0 QK fixture geometry");

    Capture capture;
    capture.path = directory / (stem.str() + ".bin");
    capture.phase = phase;
    capture.source_capacity = source_capacity;
    capture.rows = metadata_size(metadata, "query_rows");
    capture.visible = metadata_size(metadata, "visible_context");
    require((phase == "prefill" && capture.rows == 32 && capture.visible == 32) ||
        (phase == "decode" && capture.rows == 1 && capture.visible == 33),
        "unexpected captured Q/K extent");

    std::ifstream input(capture.path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open boundary capture: " + capture.path.string());
    const size_t values = head_dimension * capture.visible * kv_heads;
    const size_t probabilities = capture.visible * capture.rows * query_heads;
    const size_t outputs = head_dimension * capture.rows * query_heads;
    const size_t scores = capture.visible * capture.rows * query_heads;
    skip_bytes(input, values * sizeof(uint16_t), "V_F16");
    skip_bytes(input, probabilities * sizeof(float), "P_F32");
    skip_bytes(input, capture.rows * sizeof(int32_t), "positions_I32");
    skip_bytes(input, 3 * outputs * sizeof(float), "AV outputs and oracle");
    capture.keys_f16 = read_elements<uint16_t>(input, values, "K_F16");
    capture.query_f32 = read_elements<float>(input, head_dimension * query_heads * capture.rows, "Q_F32");
    skip_bytes(input, 4 * scores * sizeof(float), "QK and softmax reference sections");
    char trailing = 0;
    input.read(&trailing, 1);
    require(input.eof(), "capture contains unrecognized trailing bytes");
    require(std::all_of(capture.query_f32.begin(), capture.query_f32.end(),
        [](float x) { return std::isfinite(x); }), "Q capture contains non-finite values");
    return capture;
}

std::string dispatch_family(int cc, size_t capacity, size_t rows) {
    // Mirrors the pinned GGML CUDA routing predicates for F16 K and F32 Q:
    // MMVF precedes MMF; MMF is disabled for non-mul_mat_id batches >16 columns.
    const bool src0_small = capacity <= 512;
    const bool mmvf = cc >= 80 ? (src0_small && rows == 1) :
        (cc >= 70 && src0_small && rows <= 3);
    if (mmvf) return "MMVF";
    const bool mmf_alignment = head_dimension % 64 == 0;
    const bool mmf_hardware = cc >= 70;
    const bool mmf_rows = capacity % 32 == 0;
    const bool mmf_columns = rows <= 16;
    if (mmf_alignment && mmf_hardware && mmf_rows && mmf_columns) return "MMF";
    return "cuBLAS";
}

Tensor make_tensor(std::vector<float> values, std::vector<size_t> shape) {
    Tensor tensor;
    tensor.type = ScalarType::F32;
    tensor.shape = std::move(shape);
    tensor.f32 = std::move(values);
    return tensor;
}

std::vector<float> readback_logical_scores(const std::vector<float> & physical,
        size_t capacity, size_t rows, size_t visible) {
    std::vector<float> logical(rows * query_heads * visible);
    for (size_t row = 0; row < rows; ++row) for (size_t head = 0; head < query_heads; ++head)
        for (size_t position = 0; position < visible; ++position)
            logical[(row * query_heads + head) * visible + position] =
                physical[position + capacity * (row + rows * head)];
    return logical;
}

void write_f32(const fs::path & path, const std::vector<float> & values) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create QK output fixture: " + path.string());
    output.write(reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed writing QK output fixture: " + path.string());
}

void qualify(const fs::path & directory, size_t source_capacity, size_t execution_capacity,
        const std::string & phase, int device_id, const fs::path & output_path) {
    const Capture capture = read_capture(directory, source_capacity, phase);
    require(execution_capacity >= capture.visible, "execution capacity is smaller than the captured visible extent");
    const int device_count = ggml_backend_cuda_get_device_count();
    require(device_id >= 0 && device_id < device_count, "requested CUDA device is unavailable");
    cudaDeviceProp properties{};
    const cudaError_t prop_status = cudaGetDeviceProperties(&properties, device_id);
    require(prop_status == cudaSuccess, std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(prop_status));
    const int cc = properties.major * 10 + properties.minor;
    const std::string family = dispatch_family(cc, execution_capacity, capture.rows);

    const auto decoded_keys = decode_ggml_rows(GGML_TYPE_F16, capture.keys_f16.data(),
        capture.keys_f16.size() * sizeof(uint16_t), capture.visible * kv_heads, head_dimension);
    const TensorView key_view{decoded_keys.f32.data(), ScalarType::F32,
        {capture.visible, kv_heads, head_dimension},
        {static_cast<ptrdiff_t>(kv_heads * head_dimension * sizeof(float)),
         static_cast<ptrdiff_t>(head_dimension * sizeof(float)), static_cast<ptrdiff_t>(sizeof(float))}};
    const TensorView query_view = contiguous_view(capture.query_f32.data(),
        {capture.rows, query_heads, head_dimension});
    const Tensor reference = qk_scores(query_view, key_view, Accumulation::F64, ScalarType::F32);

    ggml_backend_t backend = ggml_backend_cuda_init(device_id);
    require(backend != nullptr, "failed to initialize requested GGML CUDA backend");
    ggml_init_params init = {16 * 1024 * 1024, nullptr, true};
    ggml_context * context = ggml_init(init);
    require(context != nullptr, "failed to create GGML context");
    ggml_tensor * key = ggml_new_tensor_3d(context, GGML_TYPE_F16,
        head_dimension, execution_capacity, kv_heads);
    ggml_tensor * query = ggml_new_tensor_3d(context, GGML_TYPE_F32,
        head_dimension, capture.rows, query_heads);
    ggml_tensor * scores = ggml_mul_mat(context, key, query);
    ggml_set_name(scores, "qwen3_qk_calibration_scores");
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, scores);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    require(buffer != nullptr, "failed to allocate GGML CUDA graph tensors");

    std::vector<uint16_t> padded_keys(head_dimension * execution_capacity * kv_heads, 0);
    for (size_t position = 0; position < capture.visible; ++position)
        for (size_t head = 0; head < kv_heads; ++head)
            for (size_t d = 0; d < head_dimension; ++d)
                padded_keys[d + head_dimension * (position + execution_capacity * head)] =
                    capture.keys_f16[(position * kv_heads + head) * head_dimension + d];
    std::vector<float> ggml_query(capture.query_f32.size());
    for (size_t row = 0; row < capture.rows; ++row)
        for (size_t head = 0; head < query_heads; ++head)
            for (size_t d = 0; d < head_dimension; ++d)
                ggml_query[d + head_dimension * (row + capture.rows * head)] =
                    capture.query_f32[(row * query_heads + head) * head_dimension + d];
    ggml_backend_tensor_set(key, padded_keys.data(), 0, padded_keys.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(query, ggml_query.data(), 0, ggml_query.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
        "GGML CUDA QK graph execution failed");
    std::vector<float> physical_scores(static_cast<size_t>(ggml_nelements(scores)));
    ggml_backend_tensor_get(scores, physical_scores.data(), 0, physical_scores.size() * sizeof(float));
    const auto candidate_values = readback_logical_scores(physical_scores,
        execution_capacity, capture.rows, capture.visible);
    write_f32(output_path, candidate_values);

    const Tensor candidate = make_tensor(candidate_values, {capture.rows, query_heads, capture.visible});
    EvaluationContext evaluation_context;
    evaluation_context.output_name = "qk_matmul";
    evaluation_context.candidate_identity = "diagnostic-only:ggml-cuda-qk-" + family;
    evaluation_context.qualification_run_identity = "qwen3-qk-capture-replay:v1:source-cap" +
        std::to_string(source_capacity) + ":exec-cap" + std::to_string(execution_capacity) +
        ":phase=" + phase + ":cuda" + std::to_string(device_id);
    evaluation_context.model_identity = model_identity;
    evaluation_context.backend_family = "GGML_CUDA";
    evaluation_context.implementation_identity = "ggml-cuda-qk-" + family;
    evaluation_context.device_family = "CUDA";
    std::ostringstream device_identity;
    device_identity << "CUDA" << device_id << ':' << properties.name << '@'
        << properties.pciBusID << ':' << properties.pciDeviceID;
    evaluation_context.device_identities = {device_identity.str()};
    evaluation_context.device_sm_versions = {static_cast<uint32_t>(cc)};
    evaluation_context.placement_identity = placement;
    evaluation_context.phase = phase;
    evaluation_context.execution_topology = "captured-qk-input-physical-capacity-replay";
    evaluation_context.fixture_identity = "qwen3-layer0-" + phase + "-source-cap" +
        std::to_string(source_capacity) + "-execution-cap" + std::to_string(execution_capacity);
    evaluation_context.input_identity = "qwen3-native-av-boundary-capture:" +
        capture.path.parent_path().filename().string() + ":cap" + std::to_string(source_capacity);
    evaluation_context.output_dtype = "F32";
    evaluation_context.input_dtype = "Q=F32;K=F16";
    evaluation_context.output_shape = {capture.rows, query_heads, capture.visible};
    evaluation_context.capacity = static_cast<uint32_t>(execution_capacity);
    evaluation_context.context_length = static_cast<uint32_t>(capture.visible);
    evaluation_context.rows = static_cast<uint32_t>(capture.rows);
    evaluation_context.logical_inputs_equivalent = true;

    ReferenceProvenance reference_provenance;
    reference_provenance.accumulation_precision = "binary64-product-and-left-to-right-sum";
    reference_provenance.input_representation = "Q=F32 captured post-RoPE;K=F16 captured cache decoded by pinned GGML trait";
    reference_provenance.output_representation = "F32-rounded-once";
    reference_provenance.operation_parameters = {
        {"head_dimension", "128"}, {"query_heads", "40"}, {"kv_heads", "8"},
        {"gqa_mapping", "floor(query_head/5)"}, {"attention_scale_applied", "false;applied by softmax"},
        {"dispatch_family", family}, {"dispatch_provenance", "pinned-source-predicate-replay"},
        {"physical_key_capacity", std::to_string(execution_capacity)},
        {"active_key_rows", std::to_string(capture.visible)},
        {"padded_key_rows", "zero-filled"}, {"ggml_commit", ggml_commit},
        {"cuda_device", properties.name}, {"cuda_compute_capability", std::to_string(cc)}};
    reference_provenance.replayed_input_data = true;
    const NumericalEvaluation evaluation = evaluate_reference_comparison(
        "vbuf.qk_matmul.fp64.operation_accuracy", 1, reference, candidate,
        evaluation_context, reference_provenance);
    require(evaluation.status == EvaluationStatus::NotTested &&
        evaluation.contract_status == ContractStatus::NeedsCalibration && evaluation.replayed_metrics_only,
        "QK observation unexpectedly acquired qualification authority: " + evaluation_json(evaluation));
    std::cout << evaluation_json(evaluation) << '\n';
    std::cout << "qwen3_cuda_qk_calibration device=" << device_id << " cc=" << cc
        << " phase=" << phase << " source_capacity=" << source_capacity
        << " execution_capacity=" << execution_capacity << " rows=" << capture.rows
        << " visible=" << capture.visible << " dispatch=" << family
        << " output=" << output_path.string() << " authorizing=false\n";

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        require(argc == 7, "usage: vbuf_qwen3_cuda_qk_calibration CAPTURE_DIRECTORY "
            "SOURCE_CAPACITY EXECUTION_CAPACITY prefill|decode CUDA_DEVICE OUTPUT_F32");
        const fs::path directory(argv[1]);
        const size_t source_capacity = static_cast<size_t>(std::stoull(argv[2]));
        const size_t execution_capacity = static_cast<size_t>(std::stoull(argv[3]));
        const std::string phase(argv[4]);
        const int device_id = std::stoi(argv[5]);
        require(phase == "prefill" || phase == "decode", "phase must be prefill or decode");
        require(fs::exists(directory), "capture directory does not exist: " + directory.string());
        qualify(directory, source_capacity, execution_capacity, phase, device_id, argv[6]);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vbuf_qwen3_cuda_qk_calibration=FAIL: " << error.what() << '\n';
        return 1;
    }
}

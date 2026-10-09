#pragma once

#include "vbuf_numerical_contracts.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vbuf_ggml::reference {

enum class ScalarType : uint8_t { F32, F64 };
enum class Accumulation : uint8_t { F32, F64 };

// Views address logical element [0,...,0] at data. Strides are signed byte
// strides, so transposed, sliced, broadcast, and reversed views are supported.
// The caller owns the allocation and guarantees every addressed byte is valid.
struct TensorView {
    const void * data = nullptr;
    ScalarType type = ScalarType::F32;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> byte_strides;
};

struct MaskView {
    const uint8_t * data = nullptr;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> byte_strides;
};

// A reference result owns contiguous row-major output. F32 results are rounded
// to F32 at the operation boundary; F64 results retain the binary64 value.
struct Tensor {
    ScalarType type = ScalarType::F32;
    std::vector<size_t> shape;
    std::vector<float> f32;
    std::vector<double> f64;

    size_t size() const noexcept;
    double at(size_t index) const;
};

struct Limits {
    size_t max_output_elements = 16u * 1024u * 1024u;
};

struct ReferenceProvenance {
    std::string implementation_revision = "vbuf-high-precision-reference-cpp-v1";
    std::string accumulation_precision;
    std::string input_representation;
    std::string output_representation;
    std::map<std::string, std::string> operation_parameters;
    bool replayed_input_data = false;
};

const char * implementation_revision() noexcept;
TensorView contiguous_view(const float * data, std::vector<size_t> shape);
TensorView contiguous_view(const double * data, std::vector<size_t> shape);
TensorView view(const Tensor & tensor);
vbuf_ml::numerics::NumericTensorView numeric_view(const Tensor & tensor);

// Qwen3 GQA dimensions are inferred from [rows, query_heads, head_dim] and
// [positions, kv_heads, head_dim]. Outputs raw, unscaled Q*K^T scores with
// shape [rows, query_heads, positions]; attention scaling belongs to softmax.
Tensor qk_scores(const TensorView & query, const TensorView & key,
    Accumulation accumulation = Accumulation::F64,
    ScalarType output_type = ScalarType::F32, Limits limits = {});

// Stable softmax over each row of [rows, positions]. An empty extent vector
// means all columns are logical; otherwise each row uses [0, extents[row]).
// A nonzero mask byte excludes that element. Excluded and out-of-extent outputs
// are exactly zero. Scale is explicit (Qwen3 uses 1/sqrt(head_dim)).
Tensor softmax(const TensorView & scores, double scale = 1.0,
    const std::vector<size_t> & logical_extents = {}, const MaskView * mask = nullptr,
    ScalarType output_type = ScalarType::F32, Limits limits = {});

// Values use [positions, kv_heads, head_dim], probabilities use
// [rows, query_heads, positions], and results use [rows, query_heads, head_dim].
// Per-query visible extents are optional and use the same causal bound for all
// query heads. Probability values outside the visible extent are not consumed.
Tensor attention_av(const TensorView & probabilities, const TensorView & values,
    const std::vector<size_t> & visible_extents = {},
    Accumulation accumulation = Accumulation::F64,
    ScalarType output_type = ScalarType::F32, Limits limits = {});

// Last-axis RMSNorm: x / sqrt(mean(x*x) + epsilon) * scale. Epsilon is applied
// before reciprocal square root; the reduction and products use binary64.
Tensor rms_norm(const TensorView & input, const TensorView & scale,
    double epsilon, ScalarType output_type = ScalarType::F32, Limits limits = {});

// Conventional row-major matrix product A[M,K] * B[K,N], with arbitrary
// signed byte strides and an explicit accumulation mode.
Tensor matmul(const TensorView & left, const TensorView & right,
    Accumulation accumulation = Accumulation::F64,
    ScalarType output_type = ScalarType::F32, Limits limits = {});

// Decode GGML F32/F16 and supported quantized rows via pinned GGML decoders
// (public type traits, with the explicit Q8_K row dequantizer where the public
// trait is absent). This is representation decoding only; quantization error is
// not attributed to the matmul kernel. `row_bytes` must match GGML's row size.
Tensor decode_ggml_rows(enum ggml_type type, const void * data, size_t byte_count,
    size_t rows, size_t columns, Limits limits = {});

// The checker reports hard diagnostic invariants for row-wise probabilities.
// Normalization uses 8*F32-epsilon*sqrt(active_count); this is a diagnostic
// representation check, not a numerical-accuracy threshold.
std::map<std::string, bool> probability_invariants(const TensorView & probabilities,
    const std::vector<size_t> & logical_extents, const MaskView * mask = nullptr);

// Adds revision/precision/representation/parameter provenance and evaluates
// through the existing versioned Numerical Contract System. NEEDS_CALIBRATION
// contracts remain diagnostic-only and can never become admission evidence.
vbuf_ml::numerics::NumericalEvaluation evaluate_reference_comparison(
    const std::string & contract_id, uint32_t contract_version,
    const Tensor & reference, const Tensor & candidate,
    vbuf_ml::numerics::EvaluationContext context,
    const ReferenceProvenance & provenance);

} // namespace vbuf_ggml::reference

#include "vbuf_high_precision_reference.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vbuf_ggml::reference {
namespace {
using vbuf_ml::numerics::DataType;
using vbuf_ml::numerics::EvaluationContext;
using vbuf_ml::numerics::NumericalEvaluation;

size_t checked_count(const std::vector<size_t> & shape, size_t limit) {
    if (shape.empty()) throw std::invalid_argument("tensor rank must be nonzero");
    size_t count = 1;
    for (const size_t dimension : shape) {
        if (dimension == 0) throw std::invalid_argument("tensor dimensions must be nonzero");
        if (dimension > std::numeric_limits<size_t>::max() / count)
            throw std::overflow_error("tensor element count overflow");
        count *= dimension;
        if (count > limit) throw std::length_error("reference tensor exceeds configured element limit");
    }
    return count;
}

ptrdiff_t checked_offset(const std::vector<size_t> & indices,
        const std::vector<ptrdiff_t> & strides) {
    if (indices.size() != strides.size()) throw std::invalid_argument("index/stride rank mismatch");
    ptrdiff_t result = 0;
    for (size_t axis = 0; axis < indices.size(); ++axis) {
        const ptrdiff_t stride = strides[axis];
        if (indices[axis] == 0 || stride == 0) continue;
        uintmax_t magnitude;
        if (stride < 0) magnitude = static_cast<uintmax_t>(-(stride + 1)) + 1;
        else magnitude = static_cast<uintmax_t>(stride);
        const uintmax_t index = static_cast<uintmax_t>(indices[axis]);
        const uintmax_t limit = stride < 0 ?
            static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max()) + 1 :
            static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max());
        if (index > limit / magnitude) throw std::overflow_error("tensor byte offset overflow");
        const uintmax_t product = index * magnitude;
        ptrdiff_t delta;
        if (stride < 0) {
            delta = product == limit ? std::numeric_limits<ptrdiff_t>::min() :
                -static_cast<ptrdiff_t>(product);
        } else {
            delta = static_cast<ptrdiff_t>(product);
        }
        if ((delta > 0 && result > std::numeric_limits<ptrdiff_t>::max() - delta) ||
            (delta < 0 && result < std::numeric_limits<ptrdiff_t>::min() - delta))
            throw std::overflow_error("tensor byte offset overflow");
        result += delta;
    }
    return result;
}

void validate_view(const TensorView & tensor) {
    if (tensor.data == nullptr || tensor.shape.empty() || tensor.shape.size() != tensor.byte_strides.size() ||
        (tensor.type != ScalarType::F32 && tensor.type != ScalarType::F64))
        throw std::invalid_argument("invalid tensor view");
    checked_count(tensor.shape, std::numeric_limits<size_t>::max());
    std::vector<size_t> positive_extreme(tensor.shape.size(), 0);
    std::vector<size_t> negative_extreme(tensor.shape.size(), 0);
    for (size_t axis = 0; axis < tensor.shape.size(); ++axis) {
        if (tensor.byte_strides[axis] > 0) positive_extreme[axis] = tensor.shape[axis] - 1;
        if (tensor.byte_strides[axis] < 0) negative_extreme[axis] = tensor.shape[axis] - 1;
    }
    (void) checked_offset(positive_extreme, tensor.byte_strides);
    (void) checked_offset(negative_extreme, tensor.byte_strides);
}

void validate_mask(const MaskView & mask, const std::vector<size_t> & shape) {
    if (mask.data == nullptr || mask.shape != shape || mask.byte_strides.size() != shape.size())
        throw std::invalid_argument("mask shape must match tensor shape");
    checked_count(mask.shape, std::numeric_limits<size_t>::max());
    std::vector<size_t> positive_extreme(mask.shape.size(), 0);
    std::vector<size_t> negative_extreme(mask.shape.size(), 0);
    for (size_t axis = 0; axis < mask.shape.size(); ++axis) {
        if (mask.byte_strides[axis] > 0) positive_extreme[axis] = mask.shape[axis] - 1;
        if (mask.byte_strides[axis] < 0) negative_extreme[axis] = mask.shape[axis] - 1;
    }
    (void) checked_offset(positive_extreme, mask.byte_strides);
    (void) checked_offset(negative_extreme, mask.byte_strides);
}

double load(const TensorView & tensor, const std::vector<size_t> & indices) {
    const auto * address = static_cast<const unsigned char *>(tensor.data) +
        checked_offset(indices, tensor.byte_strides);
    if (tensor.type == ScalarType::F32) {
        float value;
        std::memcpy(&value, address, sizeof(value));
        return static_cast<double>(value);
    }
    double value;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

double load_finite(const TensorView & tensor, const std::vector<size_t> & indices) {
    const double value = load(tensor, indices);
    if (!std::isfinite(value)) throw std::invalid_argument("reference operation input must be finite");
    return value;
}

double rounded_product(double left, double right) {
    volatile double product = left * right;
    return product;
}

double rounded_sum(double left, double right) {
    volatile double sum = left + right;
    return sum;
}

double rounded_difference(double left, double right) {
    volatile double difference = left - right;
    return difference;
}

bool masked(const MaskView * mask, size_t row, size_t column) {
    if (mask == nullptr) return false;
    const auto * address = mask->data + checked_offset({row, column}, mask->byte_strides);
    return *address != 0;
}

Tensor allocate(const std::vector<size_t> & shape, ScalarType type, const Limits & limits) {
    const size_t count = checked_count(shape, limits.max_output_elements);
    Tensor result;
    result.type = type;
    result.shape = shape;
    if (type == ScalarType::F32) result.f32.resize(count);
    else result.f64.resize(count);
    return result;
}

void store(Tensor & tensor, size_t index, double value) {
    if (tensor.type == ScalarType::F32) tensor.f32[index] = static_cast<float>(value);
    else tensor.f64[index] = value;
}

void validate_accumulation(Accumulation accumulation) {
    if (accumulation != Accumulation::F32 && accumulation != Accumulation::F64)
        throw std::invalid_argument("unsupported accumulation precision");
}

void validate_output_type(ScalarType type) {
    if (type != ScalarType::F32 && type != ScalarType::F64)
        throw std::invalid_argument("unsupported output type");
}

std::string scalar_type_name(ScalarType type) {
    return type == ScalarType::F32 ? "F32" : "F64";
}

} // namespace

size_t Tensor::size() const noexcept { return type == ScalarType::F32 ? f32.size() : f64.size(); }

double Tensor::at(size_t index) const {
    if (type == ScalarType::F32) return f32.at(index);
    return f64.at(index);
}

const char * implementation_revision() noexcept { return "vbuf-high-precision-reference-cpp-v1"; }

TensorView contiguous_view(const float * data, std::vector<size_t> shape) {
    if (data == nullptr) throw std::invalid_argument("null F32 tensor data");
    std::vector<ptrdiff_t> strides(shape.size());
    size_t stride = sizeof(float);
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("contiguous tensor stride overflow");
        strides[axis] = static_cast<ptrdiff_t>(stride);
        if (shape[axis] != 0 && stride > std::numeric_limits<size_t>::max() / shape[axis])
            throw std::overflow_error("contiguous tensor stride overflow");
        stride *= shape[axis];
    }
    TensorView result{data, ScalarType::F32, std::move(shape), std::move(strides)};
    validate_view(result);
    return result;
}

TensorView contiguous_view(const double * data, std::vector<size_t> shape) {
    if (data == nullptr) throw std::invalid_argument("null F64 tensor data");
    std::vector<ptrdiff_t> strides(shape.size());
    size_t stride = sizeof(double);
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("contiguous tensor stride overflow");
        strides[axis] = static_cast<ptrdiff_t>(stride);
        if (shape[axis] != 0 && stride > std::numeric_limits<size_t>::max() / shape[axis])
            throw std::overflow_error("contiguous tensor stride overflow");
        stride *= shape[axis];
    }
    TensorView result{data, ScalarType::F64, std::move(shape), std::move(strides)};
    validate_view(result);
    return result;
}

TensorView view(const Tensor & tensor) {
    const auto shape = tensor.shape;
    if (tensor.type == ScalarType::F32) return contiguous_view(tensor.f32.data(), shape);
    return contiguous_view(tensor.f64.data(), shape);
}

vbuf_ml::numerics::NumericTensorView numeric_view(const Tensor & tensor) {
    std::vector<uint64_t> shape;
    shape.reserve(tensor.shape.size());
    for (const size_t dimension : tensor.shape) shape.push_back(static_cast<uint64_t>(dimension));
    if (tensor.type == ScalarType::F32)
        return {tensor.f32.data(), tensor.f32.size(), std::move(shape), DataType::F32};
    return {tensor.f64.data(), tensor.f64.size(), std::move(shape), DataType::F64};
}

Tensor qk_scores(const TensorView & query, const TensorView & key,
        Accumulation accumulation, ScalarType output_type, Limits limits) {
    validate_view(query);
    validate_view(key);
    validate_accumulation(accumulation);
    validate_output_type(output_type);
    if (query.shape.size() != 3 || key.shape.size() != 3 ||
        query.shape[2] != key.shape[2] || query.shape[1] == 0 || key.shape[1] == 0 ||
        query.shape[1] % key.shape[1] != 0)
        throw std::invalid_argument("QK expects [rows,q_heads,dim] and [positions,kv_heads,dim] with integral GQA");
    const size_t rows = query.shape[0], q_heads = query.shape[1], dimension = query.shape[2];
    const size_t positions = key.shape[0], kv_heads = key.shape[1];
    Tensor result = allocate({rows, q_heads, positions}, output_type, limits);
    for (size_t row = 0; row < rows; ++row) for (size_t head = 0; head < q_heads; ++head) {
        const size_t kv_head = head / (q_heads / kv_heads);
        for (size_t position = 0; position < positions; ++position) {
            double sum;
            if (accumulation == Accumulation::F64) {
                sum = 0.0;
                for (size_t d = 0; d < dimension; ++d)
                    sum = rounded_sum(sum, rounded_product(load_finite(query, {row, head, d}),
                        load_finite(key, {position, kv_head, d})));
            } else {
                float fsum = 0.0f;
                for (size_t d = 0; d < dimension; ++d) {
                    volatile float product = static_cast<float>(load_finite(query, {row, head, d})) *
                        static_cast<float>(load_finite(key, {position, kv_head, d}));
                    volatile float next = fsum + product;
                    fsum = next;
                }
                sum = fsum;
            }
            store(result, (row * q_heads + head) * positions + position, sum);
        }
    }
    return result;
}

Tensor softmax(const TensorView & scores, double scale, const std::vector<size_t> & logical_extents,
        const MaskView * mask, ScalarType output_type, Limits limits) {
    validate_view(scores);
    validate_output_type(output_type);
    if (scores.shape.size() != 2) throw std::invalid_argument("softmax expects a [rows,positions] tensor");
    if (!std::isfinite(scale)) throw std::invalid_argument("softmax scale must be finite");
    const size_t rows = scores.shape[0], columns = scores.shape[1];
    if (!logical_extents.empty() && logical_extents.size() != rows)
        throw std::invalid_argument("softmax logical extent count must equal row count");
    if (mask != nullptr) validate_mask(*mask, scores.shape);
    Tensor result = allocate(scores.shape, output_type, limits);
    std::vector<size_t> active;
    for (size_t row = 0; row < rows; ++row) {
        const size_t extent = logical_extents.empty() ? columns : logical_extents[row];
        if (extent > columns) throw std::invalid_argument("softmax logical extent exceeds physical columns");
        active.clear();
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t column = 0; column < extent; ++column) {
            if (masked(mask, row, column)) continue;
            const double score = load(scores, {row, column});
            if (!std::isfinite(score)) throw std::invalid_argument("unmasked softmax scores must be finite");
            const double scaled = rounded_product(score, scale);
            if (!std::isfinite(scaled)) throw std::overflow_error("scaled softmax score is non-finite");
            maximum = std::max(maximum, scaled);
            active.push_back(column);
        }
        if (active.empty()) throw std::invalid_argument("softmax row has no unmasked logical positions");
        double denominator = 0.0;
        std::vector<double> exponentials;
        exponentials.reserve(active.size());
        for (const size_t column : active) {
            const double scaled = rounded_product(load(scores, {row, column}), scale);
            const double exponential = std::exp(rounded_difference(scaled, maximum));
            exponentials.push_back(exponential);
            denominator = rounded_sum(denominator, exponential);
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator))
            throw std::overflow_error("softmax normalization sum is invalid");
        for (size_t index = 0; index < active.size(); ++index)
            store(result, row * columns + active[index], exponentials[index] / denominator);
        for (size_t column = 0; column < columns; ++column)
            if (column >= extent || masked(mask, row, column)) store(result, row * columns + column, 0.0);
    }
    return result;
}

Tensor attention_av(const TensorView & probabilities, const TensorView & values,
        const std::vector<size_t> & visible_extents, Accumulation accumulation,
        ScalarType output_type, Limits limits) {
    validate_view(probabilities);
    validate_view(values);
    validate_accumulation(accumulation);
    validate_output_type(output_type);
    if (probabilities.shape.size() != 3 || values.shape.size() != 3 ||
        probabilities.shape[2] != values.shape[0] || values.shape[1] == 0 ||
        probabilities.shape[1] == 0 || probabilities.shape[1] % values.shape[1] != 0)
        throw std::invalid_argument("AV expects probabilities [rows,q_heads,positions] and values [positions,kv_heads,dim]");
    const size_t rows = probabilities.shape[0], q_heads = probabilities.shape[1];
    const size_t positions = probabilities.shape[2], kv_heads = values.shape[1], dimension = values.shape[2];
    if (!visible_extents.empty() && visible_extents.size() != rows)
        throw std::invalid_argument("AV visible extent count must equal query row count");
    Tensor result = allocate({rows, q_heads, dimension}, output_type, limits);
    for (size_t row = 0; row < rows; ++row) {
        const size_t extent = visible_extents.empty() ? positions : visible_extents[row];
        if (extent > positions) throw std::invalid_argument("AV visible extent exceeds positions");
        for (size_t head = 0; head < q_heads; ++head) {
            const size_t kv_head = head / (q_heads / kv_heads);
            for (size_t d = 0; d < dimension; ++d) {
                double sum = 0.0;
                float fsum = 0.0f;
                for (size_t position = 0; position < extent; ++position) {
                    const double probability = load_finite(probabilities, {row, head, position});
                    const double value = load_finite(values, {position, kv_head, d});
                    if (accumulation == Accumulation::F64)
                        sum = rounded_sum(sum, rounded_product(probability, value));
                    else {
                        volatile float product = static_cast<float>(probability) * static_cast<float>(value);
                        volatile float next = fsum + product;
                        fsum = next;
                    }
                }
                store(result, (row * q_heads + head) * dimension + d,
                    accumulation == Accumulation::F64 ? sum : static_cast<double>(fsum));
            }
        }
    }
    return result;
}

Tensor rms_norm(const TensorView & input, const TensorView & scale, double epsilon,
        ScalarType output_type, Limits limits) {
    validate_view(input);
    validate_view(scale);
    validate_output_type(output_type);
    if (input.shape.size() == 0 || scale.shape.size() != 1 ||
        scale.shape[0] != input.shape.back())
        throw std::invalid_argument("RMSNorm scale must be a vector matching the last input axis");
    if (!(epsilon > 0.0) || !std::isfinite(epsilon))
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    Tensor result = allocate(input.shape, output_type, limits);
    const size_t dimension = input.shape.back();
    size_t rows = 1;
    for (size_t axis = 0; axis + 1 < input.shape.size(); ++axis) {
        if (input.shape[axis] > std::numeric_limits<size_t>::max() / rows)
            throw std::overflow_error("RMSNorm row count overflow");
        rows *= input.shape[axis];
    }
    std::vector<size_t> prefix(input.shape.size() - 1, 0);
    for (size_t row = 0; row < rows; ++row) {
        size_t remaining = row;
        for (size_t axis = prefix.size(); axis-- > 0;) {
            prefix[axis] = remaining % input.shape[axis];
            remaining /= input.shape[axis];
        }
        double square_sum = 0.0;
        for (size_t d = 0; d < dimension; ++d) {
            auto index = prefix;
            index.push_back(d);
            const double value = load_finite(input, index);
            square_sum = rounded_sum(square_sum, rounded_product(value, value));
        }
        const double inverse_rms = 1.0 / std::sqrt(square_sum / static_cast<double>(dimension) + epsilon);
        for (size_t d = 0; d < dimension; ++d) {
            auto index = prefix;
            index.push_back(d);
            const double normalized = rounded_product(
                rounded_product(load_finite(input, index), inverse_rms), load_finite(scale, {d}));
            store(result, row * dimension + d, normalized);
        }
    }
    return result;
}

Tensor matmul(const TensorView & left, const TensorView & right,
        Accumulation accumulation, ScalarType output_type, Limits limits) {
    validate_view(left);
    validate_view(right);
    validate_accumulation(accumulation);
    validate_output_type(output_type);
    if (left.shape.size() != 2 || right.shape.size() != 2 || left.shape[1] != right.shape[0])
        throw std::invalid_argument("matmul expects compatible [M,K] and [K,N] tensors");
    const size_t m = left.shape[0], k = left.shape[1], n = right.shape[1];
    Tensor result = allocate({m, n}, output_type, limits);
    for (size_t i = 0; i < m; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        float fsum = 0.0f;
        for (size_t inner = 0; inner < k; ++inner) {
            const double a = load_finite(left, {i, inner});
            const double b = load_finite(right, {inner, j});
            if (accumulation == Accumulation::F64) sum = rounded_sum(sum, rounded_product(a, b));
            else {
                volatile float product = static_cast<float>(a) * static_cast<float>(b);
                volatile float next = fsum + product;
                fsum = next;
            }
        }
        store(result, i * n + j, accumulation == Accumulation::F64 ? sum : static_cast<double>(fsum));
    }
    return result;
}

Tensor decode_ggml_rows(enum ggml_type type, const void * data, size_t byte_count,
        size_t rows, size_t columns, Limits limits) {
    if (type < 0 || type >= GGML_TYPE_COUNT || data == nullptr || rows == 0 || columns == 0)
        throw std::invalid_argument("invalid GGML row decode request");
    const auto * traits = ggml_get_type_traits(type);
    const bool f32_copy = type == GGML_TYPE_F32;
    const bool q8k_fallback = type == GGML_TYPE_Q8_K;
    if (traits == nullptr || traits->blck_size <= 0 ||
        (!f32_copy && !q8k_fallback && traits->to_float == nullptr))
        throw std::invalid_argument("GGML type has no supported to-float decoder");
    if (columns > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("GGML row width exceeds the upstream row-size API range");
    if (columns % static_cast<size_t>(traits->blck_size) != 0)
        throw std::invalid_argument("GGML row width is not divisible by the type block size");
    const size_t row_bytes = ggml_row_size(type, static_cast<int64_t>(columns));
    if (row_bytes == 0 || rows > std::numeric_limits<size_t>::max() / row_bytes ||
        rows * row_bytes != byte_count)
        throw std::invalid_argument("GGML payload byte count does not match exact row geometry");
    Tensor result = allocate({rows, columns}, ScalarType::F32, limits);
    const auto * input = static_cast<const unsigned char *>(data);
    for (size_t row = 0; row < rows; ++row) {
        const void * row_data = input + row * row_bytes;
        if (!ggml_validate_row_data(type, row_data, row_bytes))
            throw std::invalid_argument("GGML row data failed upstream validation");
        if (f32_copy)
            std::memcpy(result.f32.data() + row * columns, row_data, row_bytes);
        else if (q8k_fallback)
            dequantize_row_q8_K(static_cast<const block_q8_K *>(row_data),
                result.f32.data() + row * columns, static_cast<int64_t>(columns));
        else
            traits->to_float(row_data, result.f32.data() + row * columns, static_cast<int64_t>(columns));
        for (size_t column = 0; column < columns; ++column)
            if (!std::isfinite(result.f32[row * columns + column]))
                throw std::invalid_argument("GGML decoder produced a non-finite value");
    }
    return result;
}

std::map<std::string, bool> probability_invariants(const TensorView & probabilities,
        const std::vector<size_t> & logical_extents, const MaskView * mask) {
    validate_view(probabilities);
    if (probabilities.shape.size() != 2)
        throw std::invalid_argument("probability invariant checker expects [rows,positions]");
    const size_t rows = probabilities.shape[0], columns = probabilities.shape[1];
    if (!logical_extents.empty() && logical_extents.size() != rows)
        throw std::invalid_argument("probability logical extent count must equal row count");
    if (mask != nullptr) validate_mask(*mask, probabilities.shape);
    bool in_range = true, normalized = true, excluded_zero = true;
    for (size_t row = 0; row < rows; ++row) {
        const size_t extent = logical_extents.empty() ? columns : logical_extents[row];
        if (extent > columns) throw std::invalid_argument("probability logical extent exceeds columns");
        double sum = 0.0;
        for (size_t column = 0; column < columns; ++column) {
            const double value = load(probabilities, {row, column});
            if (!std::isfinite(value)) {
                in_range = false;
                normalized = false;
                if (column >= extent || masked(mask, row, column)) excluded_zero = false;
                continue;
            }
            const bool excluded = column >= extent || masked(mask, row, column);
            if (excluded) {
                if (value != 0.0) excluded_zero = false;
            } else {
                if (value < 0.0 || value > 1.0) in_range = false;
                sum += value;
            }
        }
        size_t active_count = 0;
        for (size_t column = 0; column < extent; ++column)
            if (!masked(mask, row, column)) ++active_count;
        const double normalization_bound = 8.0 * std::numeric_limits<float>::epsilon() *
            std::sqrt(static_cast<double>(std::max<size_t>(1, active_count)));
        if (active_count == 0 || std::abs(sum - 1.0) > normalization_bound) normalized = false;
    }
    return {{"probabilities_in_unit_interval", in_range},
        {"probability_rows_normalized", normalized}, {"masked_positions_zero", excluded_zero}};
}

NumericalEvaluation evaluate_reference_comparison(const std::string & contract_id,
        uint32_t contract_version, const Tensor & reference, const Tensor & candidate,
        EvaluationContext context, const ReferenceProvenance & provenance) {
    if (provenance.implementation_revision != implementation_revision() ||
        provenance.accumulation_precision.empty() ||
        provenance.input_representation.empty() || provenance.output_representation.empty() ||
        provenance.operation_parameters.empty())
        throw std::invalid_argument("reference evaluation requires complete implementation and operation provenance");
    const auto * contract = vbuf_ml::numerics::find_contract(contract_id, contract_version);
    if (contract != nullptr) {
        context.operation = contract->operation;
        context.reference_kind = contract->reference_kind;
        context.reference_identity = contract->reference_identity;
    }
    context.reference_implementation_revision = provenance.implementation_revision;
    context.reference_accumulation_precision = provenance.accumulation_precision;
    context.reference_input_representation = provenance.input_representation;
    context.reference_output_representation = provenance.output_representation;
    context.reference_operation_parameters = provenance.operation_parameters;
    context.output_dtype = scalar_type_name(candidate.type);
    context.output_shape.clear();
    context.output_shape.reserve(candidate.shape.size());
    for (const size_t dimension : candidate.shape)
        context.output_shape.push_back(static_cast<uint64_t>(dimension));
    const auto reference_view = numeric_view(reference);
    const auto candidate_view = numeric_view(candidate);
    auto evaluation = vbuf_ml::numerics::evaluate_contract(contract_id, contract_version,
        &reference_view, &candidate_view, context);
    if (provenance.replayed_input_data)
        evaluation = vbuf_ml::numerics::mark_numerical_evaluation_replay_only(std::move(evaluation));
    return evaluation;
}

} // namespace vbuf_ggml::reference

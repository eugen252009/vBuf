#include "vbuf_prefill_batch.h"
#include "vbuf_tensor_wave.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    vbuf_ggml::PromptBatch batch{ { 11, 22, 33, 44 }, 7 };
    batch.validate();
    assert(batch.size() == 4);
    assert(batch.position(0) == 7);
    assert(batch.position(3) == 10);
    for (size_t query = 0; query < batch.size(); ++query) {
        for (size_t key = 0; key < batch.size(); ++key)
            assert(batch.causally_visible(query, key) == (key <= query));
    }

    // The existing GGML-backed executor consumes a hidden-by-token matrix in
    // one operation; this is the backend capability used by batched FFN work.
    const std::vector<float> identity{ 1.0f, 0.0f, 0.0f, 1.0f };
    const std::vector<float> input_values{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const uint64_t weight_dimensions[2]{ 2, 2 };
    const uint64_t input_dimensions[2]{ 2, 3 };
    const vbuf_ggml::PersistentTensorRef weight{
        7, "contract.weight", { 0, 2, weight_dimensions,
            reinterpret_cast<const uint8_t *>(identity.data()), identity.size() * sizeof(float) }, 0 };
    vbuf_ggml::TensorDependencyExecutor executor;
    const uint32_t input = executor.add_input("batch_input");
    const uint32_t persistent = executor.add_persistent(weight);
    const uint32_t output = executor.add_value("batch_output", true);
    executor.set_external_output(output);
    executor.add_operation({ "batched_matmul", vbuf_ggml::TensorWaveOpKind::MulMat,
        { { vbuf_ggml::TensorWaveRef::Kind::Persistent, persistent },
          { vbuf_ggml::TensorWaveRef::Kind::Value, input } }, output });
    const vbuf_ggml::VbufTensorView input_view{ 0, 2, input_dimensions,
        reinterpret_cast<const uint8_t *>(input_values.data()), input_values.size() * sizeof(float) };
    std::vector<uint8_t> output_bytes;
    std::vector<int64_t> output_shape;
    vbuf_ggml::TensorWaveReport report;
    const auto provider = [](const vbuf_ggml::VbufTensorView & view) {
        return vbuf_ggml::VbufBorrowedStorage{ view.payload, view.payload_len, 0, {} };
    };
    assert(executor.execute(input_view, provider, &output_bytes, &output_shape, &report) ==
        vbuf_ggml::AdapterError::None);
    assert(output_shape.size() >= 2 && output_shape[0] == 2 && output_shape[1] == 3);
    assert(output_bytes.size() == input_values.size() * sizeof(float));
    assert(std::memcmp(output_bytes.data(), input_values.data(), output_bytes.size()) == 0);
    return 0;
}

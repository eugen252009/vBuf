#include "vbuf_tensor_wave.h"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    vbuf_ggml::TensorDependencyExecutor executor;
    const uint64_t dimensions[] = { 1, 1 };
    const uint8_t payload[sizeof(float)] = {};
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, payload, sizeof(payload) };
    const uint32_t input = executor.add_input("input");
    const uint32_t unused = executor.add_persistent({ 7, "unused", view });
    const uint32_t output = executor.add_value("output", true);
    executor.set_external_output(output);
    executor.add_operation({ "identity_contract", vbuf_ggml::TensorWaveOpKind::MulMat,
        { { vbuf_ggml::TensorWaveRef::Kind::Value, input },
          { vbuf_ggml::TensorWaveRef::Kind::Value, input } }, output });

    std::vector<uint8_t> bytes;
    std::vector<int64_t> shape;
    vbuf_ggml::TensorWaveReport report;
    const auto provider = [](const vbuf_ggml::VbufTensorView &) {
        return vbuf_ggml::VbufBorrowedStorage{};
    };
    const vbuf_ggml::AdapterError error = executor.execute(
        view, provider, &bytes, &shape, &report);
    assert(unused == 0);
    assert(error == vbuf_ggml::AdapterError::InvalidArgument);
    return 0;
}

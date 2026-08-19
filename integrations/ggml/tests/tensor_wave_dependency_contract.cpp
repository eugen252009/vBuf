#include "vbuf_materializer.h"
#include "vbuf_residency.h"
#include "vbuf_tensor_wave.h"

#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>

namespace {

class FixtureSource final : public vbuf_ggml::RangeSource {
public:
    FixtureSource(const uint8_t * bytes, uint64_t size, bool fail = false)
        : bytes_(bytes), size_(size), fail_(fail) {}

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        vbuf_ggml::RangeReadResult * result) override {
        ++reads_;
        if (result != nullptr) {
            result->requested_offset = offset;
            result->requested_length = length;
            result->source_id = "fixture";
        }
        if (fail_ || destination == nullptr || offset > size_ || length > size_ - offset) {
            if (result != nullptr) result->error = "fixture failure";
            return false;
        }
        std::memcpy(destination, bytes_ + offset, length);
        if (result != nullptr) {
            result->returned_bytes = length;
            result->status_code = 200;
        }
        return true;
    }

    uint32_t reads() const { return reads_; }

private:
    const uint8_t * bytes_;
    uint64_t size_;
    bool fail_;
    uint32_t reads_ = 0;
};

struct Fixture {
    std::array<float, 1> weight{{ 3.0f }};
    std::array<float, 1> input{{ 2.0f }};
    uint64_t dimensions[2]{ 1, 1 };
    vbuf_ggml::PersistentTensorRef tensor() const {
        return { 7, "embedding.weight", { 0, 2, dimensions, nullptr, sizeof(weight) }, 0 };
    }
};

vbuf_ggml::TensorDependencyExecutor make_graph(const vbuf_ggml::PersistentTensorRef & tensor) {
    vbuf_ggml::TensorDependencyExecutor executor;
    const uint32_t input = executor.add_input("input");
    const uint32_t weight = executor.add_persistent(tensor);
    const uint32_t output = executor.add_value("output", true);
    executor.set_external_output(output);
    executor.add_operation({ "multiply", vbuf_ggml::TensorWaveOpKind::MulMat,
        { { vbuf_ggml::TensorWaveRef::Kind::Persistent, weight },
          { vbuf_ggml::TensorWaveRef::Kind::Value, input } }, output });
    return executor;
}

vbuf_ggml::AdapterError execute(vbuf_ggml::TensorDependencyExecutor & executor,
    const Fixture & fixture, vbuf_ggml::TensorMaterializer * materializer,
    std::string * detail = nullptr) {
    const vbuf_ggml::VbufTensorView input{ 0, 2, fixture.dimensions,
        reinterpret_cast<const uint8_t *>(fixture.input.data()), sizeof(fixture.input) };
    std::vector<uint8_t> output;
    std::vector<int64_t> shape;
    vbuf_ggml::TensorWaveReport report;
    const auto provider = [](const vbuf_ggml::VbufTensorView &) {
        return vbuf_ggml::VbufBorrowedStorage{};
    };
    return executor.execute(input, provider, &output, &shape, &report, detail, {}, materializer);
}

} // namespace

int main() {
    Fixture fixture;
    const auto tensor = fixture.tensor();

    auto source = std::make_shared<FixtureSource>(
        reinterpret_cast<const uint8_t *>(fixture.weight.data()), sizeof(fixture.weight));
    auto backing = std::make_shared<vbuf_ggml::LocalVbufRangeMaterializer>(source);
    auto residency = std::make_shared<vbuf_ggml::TensorResidencyStore>(sizeof(fixture.weight));
    vbuf_ggml::ResidentTensorMaterializer materializer(backing, residency);
    auto executor = make_graph(tensor);

    // A dependency with no prefetch request is requested and waited at consume time.
    assert(execute(executor, fixture, &materializer) == vbuf_ggml::AdapterError::None);
    assert(source->reads() == 1);

    // A resident payload is immediately ready and does not re-read the source.
    assert(execute(executor, fixture, &materializer) == vbuf_ggml::AdapterError::None);
    assert(source->reads() == 1);

    // A failed producer becomes a deterministic consumer error.
    auto failed_source = std::make_shared<FixtureSource>(
        reinterpret_cast<const uint8_t *>(fixture.weight.data()), sizeof(fixture.weight), true);
    auto failed_backing = std::make_shared<vbuf_ggml::LocalVbufRangeMaterializer>(failed_source);
    auto failed_residency = std::make_shared<vbuf_ggml::TensorResidencyStore>(sizeof(fixture.weight));
    vbuf_ggml::ResidentTensorMaterializer failed_materializer(failed_backing, failed_residency);
    std::string detail;
    assert(execute(executor, fixture, &failed_materializer, &detail) ==
        vbuf_ggml::AdapterError::InvalidArgument);
    assert(detail == "persistent tensor is not ready: FAILED");
    assert(failed_residency->active_lease_count() == 0);

    // A borrowed resident payload prevents eviction until its materializer lease is released.
    assert(materializer.request(0, tensor, sizeof(fixture.weight)));
    assert(materializer.wait(0) == vbuf_ggml::MaterializationState::Ready);
    const auto borrowed = materializer.obtain_ready_tensor(0);
    assert(borrowed.has_value());
    assert(materializer.request(0, tensor, sizeof(fixture.weight)));
    assert(std::memcmp(borrowed->view.payload, fixture.weight.data(), sizeof(fixture.weight)) == 0);
    assert(!residency->evict(0));
    materializer.release(0);
    assert(residency->evict(0));
    return 0;
}

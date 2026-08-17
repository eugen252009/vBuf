#include "vbuf_materializer.h"

#include <cassert>
#include <cstdint>
#include <cstring>

int main() {
    const uint64_t dimensions[] = { 4, 1 };
    const uint8_t source[] = { 1, 2, 3, 4 };
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, source, sizeof(source) };
    const vbuf_ggml::PersistentTensorRef tensor{ 9, "tensor", view };
    vbuf_ggml::LocalVbufRangeMaterializer materializer;

    assert(materializer.state(0) == vbuf_ggml::MaterializationState::NotRequested);
    assert(!materializer.request(0, tensor, sizeof(source) - 1));
    assert(materializer.request(0, tensor, sizeof(source)));
    assert(materializer.state(0) == vbuf_ggml::MaterializationState::InFlight ||
        materializer.state(0) == vbuf_ggml::MaterializationState::Ready);
    assert(materializer.wait(0) == vbuf_ggml::MaterializationState::Ready);
    const auto ready = materializer.obtain_ready_tensor(0);
    assert(ready.has_value());
    assert(ready->bytes == sizeof(source));
    assert(std::memcmp(ready->view.payload, source, sizeof(source)) == 0);
    assert(materializer.active_ready_bytes() == sizeof(source));
    materializer.release(0);
    assert(materializer.state(0) == vbuf_ggml::MaterializationState::Released);
    assert(materializer.active_inflight_bytes() == 0);
    assert(materializer.active_ready_bytes() == 0);
    assert(materializer.request(0, tensor, sizeof(source)));
    assert(materializer.wait(0) == vbuf_ggml::MaterializationState::Ready);
    materializer.release(0);
    return 0;
}

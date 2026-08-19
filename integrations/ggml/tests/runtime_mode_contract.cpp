#include "vbuf_runtime_mode.h"

#include <cassert>

int main() {
    assert(!vbuf_ggml::runs_reference_control(vbuf_ggml::RuntimeMode::NormalInference));
    assert(vbuf_ggml::runs_reference_control(vbuf_ggml::RuntimeMode::Qualification));
    return 0;
}

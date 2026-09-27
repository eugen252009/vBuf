#include "vbuf_materializer.h"
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

using namespace vbuf_ggml;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    constexpr size_t bytes = 64 * 1024;
    std::vector<uint8_t> payload(bytes, 0x5a);
    const uint64_t shape[] = {bytes / sizeof(float)};
    const PersistentTensorRef tensor{17, "page_lifetime", {0, 1, shape, payload.data(), bytes}, 0};
    auto source = std::make_shared<LocalVbufRangeSource>(payload.data(), payload.size());
    LocalVbufRangeMaterializer materializer(source);
    require(materializer.request(17, tensor, bytes), "request");
    require(materializer.wait(17) == MaterializationState::Ready, "wait");
    auto ready = materializer.obtain_ready_tensor(17);
    require(ready.has_value(), "ready");
    auto lease = ready->storage.lease;
    auto * address = const_cast<uint8_t *>(ready->storage.base);
    const long page = sysconf(_SC_PAGESIZE);
    require(page > 0 && reinterpret_cast<uintptr_t>(address) % page == 0, "page-aligned owner");
    std::vector<unsigned char> residency((bytes + page - 1) / page);
    materializer.release(17);
    ready.reset();
    require(materializer.active_ready_bytes() == 0, "ready accounting released");
    require(mincore(address, bytes, residency.data()) == 0, "live lease keeps mapping");
    require(std::memcmp(address, payload.data(), bytes) == 0, "live lease preserves bytes");
    lease.reset();
    errno = 0;
    require(mincore(address, bytes, residency.data()) == -1 && errno == ENOMEM,
        "final lease must unmap pages, not retain them in malloc arenas");
}

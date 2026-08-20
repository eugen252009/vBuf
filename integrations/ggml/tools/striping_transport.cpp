#include "vbuf_materializer.h"
#include "vbuf_striped_materializer.h"
#include "vbuf_residency.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

using vbuf_ggml::HttpRangeSource;
using vbuf_ggml::LocalVbufRangeMaterializer;
using vbuf_ggml::MaterializationState;
using vbuf_ggml::ParallelRangeMaterializer;
using vbuf_ggml::PersistentTensorRef;
using vbuf_ggml::RangeStripingPlan;
using vbuf_ggml::TensorMaterializer;
using vbuf_ggml::VbufTensorView;

uint64_t fnv1a(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

int main(int argc, char ** argv) {
    if ((argc != 6 && argc != 8 && argc != 9) || (std::string(argv[1]) != "single" &&
        std::string(argv[1]) != "striped")) {
        std::fprintf(stderr, "usage: striping_transport single offset length url local_ip\n"
            "   or: striping_transport striped offset length url_a url_b local_ip_a local_ip_b [warm]\n");
        return 2;
    }
    const uint64_t offset = std::stoull(argv[2]);
    const uint64_t length = std::stoull(argv[3]);
    const bool warm = std::string(argv[1]) == "striped" && argc == 9;
    const uint64_t dimensions[] = { 1, 1 };
    const VbufTensorView view{ 0, 2, dimensions, nullptr, length };
    const PersistentTensorRef tensor{ 0, "transport-range", view, offset };
    std::shared_ptr<TensorMaterializer> materializer;
    std::shared_ptr<ParallelRangeMaterializer> striped;
    std::shared_ptr<vbuf_ggml::TensorResidencyStore> residency;
    if (std::string(argv[1]) == "single") {
        auto source = std::make_shared<HttpRangeSource>(argv[4], argv[5]);
        materializer = std::make_shared<LocalVbufRangeMaterializer>(source);
    } else {
        auto source_a = std::make_shared<HttpRangeSource>(argv[4], argv[6]);
        auto source_b = std::make_shared<HttpRangeSource>(argv[5], argv[7]);
        const auto plan = RangeStripingPlan::two_way(offset, length, source_a, source_b, 32);
        if (!plan) {
            std::fprintf(stderr, "invalid striping plan\n");
            return 3;
        }
        striped = std::make_shared<ParallelRangeMaterializer>(*plan);
        materializer = striped;
        std::printf("plan offset=%llu length=%llu A_offset=%llu A_length=%llu B_offset=%llu B_length=%llu\n",
            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(length),
            static_cast<unsigned long long>(plan->stripes[0].source_offset),
            static_cast<unsigned long long>(plan->stripes[0].length),
            static_cast<unsigned long long>(plan->stripes[1].source_offset),
            static_cast<unsigned long long>(plan->stripes[1].length));
    }
    if (warm) {
        auto backing = materializer;
        residency = std::make_shared<vbuf_ggml::TensorResidencyStore>(length);
        materializer = std::make_shared<vbuf_ggml::ResidentTensorMaterializer>(backing, residency);
    }
    auto materialize_once = [&]() -> std::optional<vbuf_ggml::MaterializedTensor> {
        if (!materializer->request(0, tensor, length) ||
            materializer->wait(0) != MaterializationState::Ready) return std::nullopt;
        const auto result = materializer->obtain_ready_tensor(0);
        materializer->release(0);
        return result;
    };
    const auto ready = materialize_once();
    if (!ready.has_value()) {
        std::fprintf(stderr, "transport materialization failed\n");
        return 4;
    }
    std::printf("ready bytes=%llu\n", static_cast<unsigned long long>(ready->bytes));
    if (length <= 64 * 1024 * 1024) {
        std::printf("payload_hash=%016llx\n",
            static_cast<unsigned long long>(fnv1a(ready->view().payload, ready->bytes)));
    }
    if (warm) {
        const size_t cold_stripe_operations = striped->stripe_trace().size();
        const auto warm_ready = materialize_once();
        std::printf("warm_hit=%s warm_stripe_operations=%zu\n",
            warm_ready.has_value() ? "PASS" : "FAIL",
            striped->stripe_trace().size() - cold_stripe_operations);
    }
    for (const auto & event : materializer->trace()) {
        std::printf("materialization event=%s state=%s timestamp_ns=%llu first_byte_ns=%llu "
            "bytes=%llu returned=%llu source=%s local=%s remote=%s\n",
            event.event.c_str(), vbuf_ggml::materialization_state_name(event.state),
            static_cast<unsigned long long>(event.timestamp_ns),
            static_cast<unsigned long long>(event.first_byte_timestamp_ns),
            static_cast<unsigned long long>(event.bytes),
            static_cast<unsigned long long>(event.returned_bytes), event.source_id.c_str(),
            event.local_endpoint.c_str(), event.remote_endpoint.c_str());
    }
    if (striped) {
        uint64_t total = 0;
        uint64_t start = UINT64_MAX;
        uint64_t finish = 0;
        for (const auto & event : striped->stripe_trace()) {
            total += event.returned_bytes;
            start = std::min(start, event.request_start_ns);
            finish = std::max(finish, event.complete_ns);
            std::printf("stripe id=%s start_ns=%llu first_byte_ns=%llu complete_ns=%llu "
                "offset=%llu length=%llu returned=%llu source=%s local=%s remote=%s\n",
                event.stripe_id.c_str(), static_cast<unsigned long long>(event.request_start_ns),
                static_cast<unsigned long long>(event.first_byte_ns),
                static_cast<unsigned long long>(event.complete_ns),
                static_cast<unsigned long long>(event.requested_offset),
                static_cast<unsigned long long>(event.requested_length),
                static_cast<unsigned long long>(event.returned_bytes), event.source_id.c_str(),
                event.local_endpoint.c_str(), event.remote_endpoint.c_str());
        }
        std::printf("striped total_bytes=%llu wall_ns=%llu\n",
            static_cast<unsigned long long>(total),
            static_cast<unsigned long long>(finish - start));
    }
    if (residency) residency->clear();
    std::printf("resources_after_teardown=%llu\n",
        static_cast<unsigned long long>(materializer->active_inflight_bytes() +
            materializer->active_ready_bytes()));
    return 0;
}

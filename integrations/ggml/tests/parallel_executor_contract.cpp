#include "vbuf_parallel_executor.h"
#include "vbuf_weighted_merge.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

using namespace vbuf_ggml;
static void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }

int main() {
    bool rejected = false;
    try { BoundedExecutor invalid(0); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "zero workers accepted");
    BoundedExecutor pool(3);
    rejected = false;
    try { pool.run(SIZE_MAX, [](size_t) {}); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "overflowing batch accepted");
    std::mutex mutex;
    std::condition_variable done;
    size_t early = 0;
    std::vector<size_t> completion;
    std::vector<std::vector<float>> values(3);
    pool.run(3, [&](size_t rank) {
        require(cpu_execution_threads() == 0, "thread setting leaked across jobs");
        { ScopedCpuExecutionThreads setting(1); require(cpu_execution_threads() == 1, "thread budget"); }
        if (rank == 0) {
            std::unique_lock<std::mutex> lock(mutex);
            require(done.wait_for(lock, std::chrono::seconds(5), [&] { return early == 2; }), "jobs did not overlap");
        }
        values[rank] = {rank == 0 ? 1e20f : rank == 1 ? -1e20f : 3.0f};
        { std::lock_guard<std::mutex> lock(mutex); completion.push_back(rank); if (rank != 0) ++early; }
        done.notify_all();
    });
    require(completion.back() == 0 && pool.peak_workers() >= 2 && pool.peak_workers() <= 3, "bounded overlap");
    std::vector<float> output;
    std::string error;
    require(weighted_merge(values, {1, 1, 1}, &output, &error) && output == std::vector<float>{3},
        "completion order changed floating-point reduction order");
    std::atomic<size_t> visited{0};
    bool failed = false;
    try {
        pool.run(12, [&](size_t rank) {
            ++visited;
            if (rank == 2 || rank == 5) throw std::runtime_error(std::to_string(rank));
        });
    } catch (const std::runtime_error & error) { failed = std::string(error.what()) == "2"; }
    require(failed && visited == 12, "failure did not drain batch or select deterministic error");
    pool.reset_metrics();
    pool.run(1, [&](size_t) { ++visited; });
    require(visited == 13 && pool.peak_workers() == 1, "pool did not recover after failure");
    BoundedExecutor serial(1);
    visited = 0;
    try { serial.run(3, [&](size_t i) { ++visited; if (i == 0) throw std::runtime_error("serial"); }); }
    catch (const std::runtime_error &) {}
    require(visited == 3, "serial fallback must also drain its batch");
}

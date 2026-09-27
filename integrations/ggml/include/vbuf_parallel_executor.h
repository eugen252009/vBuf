#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace vbuf_ggml {

// A synchronous batch on a reusable, bounded pool. Jobs write distinct result
// slots; the caller retains payload leases and performs ordered reductions.
// All jobs are drained before return/throw, including after a job failure.
// Batches on one pool are serialized; callbacks must not re-enter that pool.
class BoundedExecutor {
public:
    explicit BoundedExecutor(size_t workers);
    ~BoundedExecutor();
    BoundedExecutor(const BoundedExecutor &) = delete;
    BoundedExecutor & operator=(const BoundedExecutor &) = delete;
    void run(size_t jobs, const std::function<void(size_t)> & function);
    size_t workers() const { return worker_count_; }
    size_t peak_workers() const { return peak_.load(); }
    void reset_metrics() { peak_ = 0; }

private:
    void worker();
    size_t worker_count_;
    std::vector<std::thread> threads_;
    std::mutex submit_mutex_, mutex_;
    std::condition_variable ready_, done_;
    bool stopping_ = false;
    size_t epoch_ = 0, remaining_ = 0, jobs_ = 0, failed_index_ = SIZE_MAX;
    std::function<void(size_t)> function_;
    std::exception_ptr error_;
    std::atomic<size_t> next_{0}, active_{0}, peak_{0};
};

// Thread-local compute budget: independent expert jobs must not each create a
// full-machine backend team. Zero preserves the backend's normal default.
int cpu_execution_threads();
class ScopedCpuExecutionThreads {
public:
    explicit ScopedCpuExecutionThreads(int threads);
    ~ScopedCpuExecutionThreads();
    ScopedCpuExecutionThreads(const ScopedCpuExecutionThreads &) = delete;
    ScopedCpuExecutionThreads & operator=(const ScopedCpuExecutionThreads &) = delete;
private:
    int previous_;
};

struct ExpertExecution {
    explicit ExpertExecution(size_t workers, int threads) : pool(workers), threads_per_expert(threads) {}
    BoundedExecutor pool;
    int threads_per_expert;
    uint64_t jobs = 0, waves = 0, peak_prepared_bytes = 0, serial_fallbacks = 0;
    void reset_metrics() { jobs = waves = peak_prepared_bytes = serial_fallbacks = 0; pool.reset_metrics(); }
};

} // namespace vbuf_ggml

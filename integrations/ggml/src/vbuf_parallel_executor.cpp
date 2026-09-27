#include "vbuf_parallel_executor.h"
#include <stdexcept>

namespace vbuf_ggml {
namespace { thread_local int execution_threads = 0; }
int cpu_execution_threads() { return execution_threads; }
ScopedCpuExecutionThreads::ScopedCpuExecutionThreads(int threads) : previous_(execution_threads) {
    if (threads < 1) throw std::invalid_argument("compute threads must be positive");
    execution_threads = threads;
}
ScopedCpuExecutionThreads::~ScopedCpuExecutionThreads() { execution_threads = previous_; }

BoundedExecutor::BoundedExecutor(size_t workers) : worker_count_(workers) {
    if (workers < 1 || workers > 64) throw std::invalid_argument("worker count must be in 1..64");
    if (workers == 1) return;
    try {
        for (size_t i = 0; i < workers; ++i) threads_.emplace_back([this] { worker(); });
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto & thread : threads_) thread.join();
        throw;
    }
}

BoundedExecutor::~BoundedExecutor() {
    std::lock_guard<std::mutex> submission(submit_mutex_);
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_all();
    for (auto & thread : threads_) thread.join();
}

void BoundedExecutor::run(size_t jobs, const std::function<void(size_t)> & function) {
    std::lock_guard<std::mutex> submission(submit_mutex_);
    if (jobs == 0) return;
    if (jobs > SIZE_MAX - worker_count_) throw std::invalid_argument("job counter would overflow");
    if (!function) throw std::invalid_argument("missing parallel job function");
    if (worker_count_ == 1) {
        peak_ = 1;
        std::exception_ptr error;
        for (size_t i = 0; i < jobs; ++i) {
            try { function(i); } catch (...) { if (!error) error = std::current_exception(); }
        }
        if (error) std::rethrow_exception(error);
        return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    function_ = function;
    jobs_ = jobs; next_ = 0; remaining_ = worker_count_;
    error_ = nullptr; failed_index_ = SIZE_MAX;
    ++epoch_;
    ready_.notify_all();
    done_.wait(lock, [&] { return remaining_ == 0; });
    function_ = {};
    if (error_) std::rethrow_exception(error_);
}

void BoundedExecutor::worker() {
    size_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopping_ || epoch_ != seen; });
        if (stopping_) return;
        seen = epoch_;
        // The function stays immutable until all workers drain this batch.
        lock.unlock();
        for (;;) {
            const size_t index = next_.fetch_add(1);
            if (index >= jobs_) break;
            const size_t active = active_.fetch_add(1) + 1;
            size_t peak = peak_.load();
            while (peak < active && !peak_.compare_exchange_weak(peak, active)) {}
            try { function_(index); }
            catch (...) {
                std::lock_guard<std::mutex> error_lock(mutex_);
                if (index < failed_index_) { failed_index_ = index; error_ = std::current_exception(); }
            }
            --active_;
        }
        lock.lock();
        if (--remaining_ == 0) done_.notify_one();
    }
}
} // namespace vbuf_ggml

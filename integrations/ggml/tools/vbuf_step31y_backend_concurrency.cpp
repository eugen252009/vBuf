#define VBUF_COMPAT_SERVER_LIBRARY_ONLY
#include "vbuf_compat_server.cpp"
#undef VBUF_COMPAT_SERVER_LIBRARY_ONLY

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <thread>

namespace {

constexpr const char * AVAILABLE_GGML_COMMIT =
    "a97123e497968f3440264c0464a7adc7c999c027";

struct Config {
    std::string semantic_model;
    std::string source_url;
    std::string prompt_a = "Say hi";
    std::string prompt_b = "Count to one";
    std::string mode;
    uint32_t blocks = 2;
    uint64_t capacity = 268435456;
    uint32_t max_new_tokens = 8;
    uint32_t warmup = 1;
    uint32_t repetitions = 3;
};

struct ProcMetrics {
    uint64_t rss_kib = 0;
    uint64_t pss_kib = 0;
    uint64_t threads = 0;
};

struct RunRecord {
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    uint64_t prompt_hash = 0;
    uint64_t token_hash = 0;
    size_t token_count = 0;
    vbuf_ggml::VbufGenerationResult result;
    vbuf_ggml::VbufGenerationSnapshot snapshot;
};

struct PairRecord {
    RunRecord a;
    RunRecord b;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    uint64_t overlap_ns = 0;
};

class StartGate {
public:
    explicit StartGate(size_t participants) : participants_(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const size_t arrived = ++arrived_;
        if (arrived == participants_) {
            open_ = true;
            condition_.notify_all();
            return;
        }
        condition_.wait(lock, [&] { return open_; });
    }

private:
    const size_t participants_;
    size_t arrived_ = 0;
    bool open_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

static std::string required_value(int * index, int argc, char ** argv) {
    if (*index + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + argv[*index]);
    return argv[++*index];
}

static Config parse_concurrency_args(int argc, char ** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--semantic-model") config.semantic_model = required_value(&index, argc, argv);
        else if (arg == "--source-url") config.source_url = required_value(&index, argc, argv);
        else if (arg == "--prompt-a") config.prompt_a = required_value(&index, argc, argv);
        else if (arg == "--prompt-b") config.prompt_b = required_value(&index, argc, argv);
        else if (arg == "--mode") config.mode = required_value(&index, argc, argv);
        else if (arg == "--blocks") config.blocks = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else if (arg == "--capacity") config.capacity = std::stoull(required_value(&index, argc, argv));
        else if (arg == "--max-new-tokens") config.max_new_tokens = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else if (arg == "--warmup") config.warmup = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else if (arg == "--repetitions") config.repetitions = static_cast<uint32_t>(std::stoul(required_value(&index, argc, argv)));
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (config.semantic_model.empty() || config.source_url.empty() ||
        (config.mode != "serial" && config.mode != "concurrent" &&
            config.mode != "shared-serial" && config.mode != "shared-concurrent") ||
        config.blocks == 0 || config.max_new_tokens == 0 || config.repetitions == 0)
        throw std::runtime_error("usage: vbuf_step31y_backend_concurrency --semantic-model PATH "
            "--source-url URL --mode serial|concurrent|shared-serial|shared-concurrent "
            "[--prompt-a TEXT --prompt-b TEXT "
            "--blocks N --capacity BYTES --max-new-tokens N --warmup N --repetitions N]");
    return config;
}

static uint64_t proc_value(const char * path, const char * key) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string label;
        uint64_t value = 0;
        fields >> label >> value;
        if (label == key) return value;
    }
    return 0;
}

static ProcMetrics read_proc_metrics() {
    ProcMetrics metrics;
    metrics.rss_kib = proc_value("/proc/self/status", "VmRSS:");
    metrics.threads = proc_value("/proc/self/status", "Threads:");
    std::ifstream smaps("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(smaps, line)) {
        std::istringstream fields(line);
        std::string label;
        uint64_t value = 0;
        fields >> label >> value;
        if (label == "Pss:") metrics.pss_kib = value;
    }
    return metrics;
}

class ProcMonitor {
public:
    void start() {
        running_.store(true);
        thread_ = std::thread([this] {
            while (running_.load()) {
                observe(read_proc_metrics());
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            observe(read_proc_metrics());
        });
    }

    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
    }

    void reset() {
        const ProcMetrics current = read_proc_metrics();
        std::lock_guard<std::mutex> lock(mutex_);
        baseline_ = current;
        peak_ = current;
    }

    ProcMetrics baseline() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return baseline_;
    }

    ProcMetrics peak() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

private:
    void observe(const ProcMetrics & current) {
        std::lock_guard<std::mutex> lock(mutex_);
        peak_.rss_kib = std::max(peak_.rss_kib, current.rss_kib);
        peak_.pss_kib = std::max(peak_.pss_kib, current.pss_kib);
        peak_.threads = std::max(peak_.threads, current.threads);
    }

    std::atomic<bool> running_{false};
    std::thread thread_;
    mutable std::mutex mutex_;
    ProcMetrics baseline_;
    ProcMetrics peak_;
};

static vbuf_ggml::VbufGenerationConfig make_generation(const Config & config,
    const std::vector<uint32_t> & prompt_tokens) {
    vbuf_ggml::VbufGenerationConfig generation;
    generation.semantic_model = config.semantic_model;
    generation.source_endpoint = config.source_url;
    generation.block_count = config.blocks;
    generation.residency_capacity = config.capacity;
    generation.max_new_tokens = config.max_new_tokens;
    generation.mode = vbuf_ggml::RuntimeMode::NormalInference;
    generation.prompt_tokens = prompt_tokens;
    generation.on_token = [](uint32_t, uint32_t) { return true; };
    generation.should_cancel = [] { return false; };
    return generation;
}

static RunRecord run_one(const Config & config, vbuf_ggml::VbufGenerationSession * session,
    const std::vector<uint32_t> & prompt_tokens, StartGate * gate) {
    RunRecord record;
    record.prompt_hash = token_hash(prompt_tokens);
    record.token_count = prompt_tokens.size();
    if (gate != nullptr) gate->wait();
    record.start_ns = steady_now_ns();
    try {
        StdoutSilencer silence;
        record.result = session->run(make_generation(config, prompt_tokens));
    } catch (const std::exception & error) {
        record.result.error = error.what();
    }
    record.end_ns = steady_now_ns();
    record.token_count = record.result.tokens.size();
    record.token_hash = token_hash(record.result.tokens);
    record.snapshot = session->snapshot();
    return record;
}

static PairRecord run_pair(const Config & config, vbuf_ggml::VbufGenerationSession * session_a,
    vbuf_ggml::VbufGenerationSession * session_b, const std::vector<uint32_t> & prompt_a,
    const std::vector<uint32_t> & prompt_b) {
    PairRecord pair;
    pair.start_ns = steady_now_ns();
    const bool concurrent = config.mode == "concurrent" || config.mode == "shared-concurrent";
    if (!concurrent) {
        pair.a = run_one(config, session_a, prompt_a, nullptr);
        pair.b = run_one(config, session_b, prompt_b, nullptr);
    } else {
        StartGate gate(2);
        std::thread a([&] { pair.a = run_one(config, session_a, prompt_a, &gate); });
        std::thread b([&] { pair.b = run_one(config, session_b, prompt_b, &gate); });
        a.join();
        b.join();
    }
    pair.end_ns = std::max(pair.a.end_ns, pair.b.end_ns);
    const uint64_t overlap_start = std::max(pair.a.start_ns, pair.b.start_ns);
    const uint64_t overlap_end = std::min(pair.a.end_ns, pair.b.end_ns);
    pair.overlap_ns = overlap_end > overlap_start ? overlap_end - overlap_start : 0;
    return pair;
}

static void print_record(const char * name, const RunRecord & record) {
    std::cout << "{\"name\":\"" << name << "\",\"start_ns\":" << record.start_ns
        << ",\"end_ns\":" << record.end_ns << ",\"duration_ns\":"
        << record.end_ns - record.start_ns << ",\"prompt_hash\":\""
        << std::hex << std::setw(16) << std::setfill('0') << record.prompt_hash << std::dec
        << "\",\"token_hash\":\"" << std::hex << std::setw(16) << std::setfill('0')
        << record.token_hash << std::dec << "\",\"token_count\":" << record.token_count
        << ",\"completed\":" << (record.result.completed ? "true" : "false")
        << ",\"cancelled\":" << (record.result.cancelled ? "true" : "false")
        << ",\"error\":" << (record.result.error.empty() ? "null" : "\"runtime_error\"")
        << ",\"source_bytes\":" << record.result.source_bytes
        << ",\"materialized_bytes\":" << record.result.materialized_bytes
        << ",\"resident_bytes_before\":" << record.result.resident_bytes_before
        << ",\"peak_resident_bytes\":" << record.result.peak_resident_bytes
        << ",\"resident_bytes_after\":" << record.result.resident_bytes_after
        << ",\"active_leases_after\":" << record.result.active_lease_count_after
        << ",\"active_lease_bytes_after\":" << record.result.active_lease_bytes_after
        << ",\"active_inflight_bytes_after\":" << record.result.active_inflight_bytes_after
        << ",\"evictions\":" << record.result.evictions
        << ",\"reacquisitions\":" << record.result.reacquisitions
        << ",\"prefill_ns\":" << record.result.prefill_ns
        << ",\"decode_ns\":" << record.result.decode_ns
        << ",\"source_requests\":" << record.snapshot.source_requests
        << ",\"source_connections\":" << record.snapshot.source_connections
        << ",\"snapshot_resident_bytes\":" << record.snapshot.resident_bytes
        << ",\"snapshot_active_generations\":" << record.snapshot.active_generations
        << "}";
}

static bool valid_record(const RunRecord & record) {
    return record.result.error.empty() && record.result.completed && !record.result.cancelled &&
        record.result.active_lease_count_after == 0 && record.result.active_lease_bytes_after == 0 &&
        record.result.active_inflight_bytes_after == 0 && record.snapshot.active_generations == 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Config config = parse_concurrency_args(argc, argv);
        const bool shared_substrate = config.mode == "shared-serial" ||
            config.mode == "shared-concurrent";
        VbufTokenizer tokenizer(config.semantic_model);
        const std::vector<Message> messages_a = {{"user", config.prompt_a}};
        const std::vector<Message> messages_b = {{"user", config.prompt_b}};
        const std::vector<uint32_t> prompt_a = tokenizer.encode_chat(messages_a);
        const std::vector<uint32_t> prompt_b = tokenizer.encode_chat(messages_b);
        if (prompt_a.empty() || prompt_b.empty()) throw std::runtime_error("empty prompt tokenization");

        std::shared_ptr<vbuf_ggml::TensorResidencyStore> shared_residency;
        if (shared_substrate) {
            shared_residency = std::make_shared<vbuf_ggml::TensorResidencyStore>(config.capacity,
                vbuf_ggml::ResidencyReplacementPolicyKind::CostAware);
        }
        auto session_a = shared_residency
            ? std::make_unique<vbuf_ggml::VbufGenerationSession>(
                config.semantic_model, config.blocks, shared_residency)
            : std::make_unique<vbuf_ggml::VbufGenerationSession>(config.semantic_model, config.blocks);
        auto session_b = shared_residency
            ? std::make_unique<vbuf_ggml::VbufGenerationSession>(
                config.semantic_model, config.blocks, shared_residency)
            : std::make_unique<vbuf_ggml::VbufGenerationSession>(config.semantic_model, config.blocks);
        ProcMonitor monitor;
        monitor.start();
        for (uint32_t index = 0; index < config.warmup; ++index) {
            const RunRecord warm_a = run_one(config, session_a.get(), prompt_a, nullptr);
            const RunRecord warm_b = run_one(config, session_b.get(), prompt_b, nullptr);
            if (!valid_record(warm_a) || !valid_record(warm_b))
                throw std::runtime_error("warmup generation failed");
        }
        monitor.reset();
        std::vector<PairRecord> pairs;
        pairs.reserve(config.repetitions);
        bool valid = true;
        uint64_t expected_a = 0;
        uint64_t expected_b = 0;
        for (uint32_t index = 0; index < config.repetitions; ++index) {
            PairRecord pair = run_pair(config, session_a.get(), session_b.get(), prompt_a, prompt_b);
            if (index == 0) {
                expected_a = pair.a.token_hash;
                expected_b = pair.b.token_hash;
            }
            valid = valid && valid_record(pair.a) && valid_record(pair.b) &&
                pair.a.token_hash == expected_a && pair.b.token_hash == expected_b;
            pairs.push_back(std::move(pair));
        }
        monitor.stop();
        const ProcMetrics baseline = monitor.baseline();
        const ProcMetrics peak = monitor.peak();
        uint64_t total_ns = 0;
        uint64_t total_overlap_ns = 0;
        for (const PairRecord & pair : pairs) {
            total_ns += pair.end_ns - pair.start_ns;
            total_overlap_ns += pair.overlap_ns;
        }
        const uint64_t positions_a = prompt_a.size() + config.max_new_tokens;
        const uint64_t positions_b = prompt_b.size() + config.max_new_tokens;
        const uint64_t kv_estimate = 4ULL * (16ULL * 192ULL + 16ULL * 128ULL) * sizeof(float) *
            (positions_a + positions_b);

        std::cout << "{\"step\":\"" << (shared_substrate ? "31Z" : "31Y")
            << "\",\"mode\":\"" << config.mode
            << "\",\"ggml_commit\":\"" << AVAILABLE_GGML_COMMIT
            << "\",\"semantic_model\":\"" << config.semantic_model
            << "\",\"source_url\":\"" << config.source_url
            << "\",\"prompt_a_hash\":\"" << std::hex << std::setw(16) << std::setfill('0')
            << token_hash(prompt_a) << "\",\"prompt_b_hash\":\"" << std::setw(16)
            << token_hash(prompt_b) << std::dec << "\",\"blocks\":" << config.blocks
            << ",\"max_new_tokens\":" << config.max_new_tokens
            << ",\"warmup_per_session\":" << config.warmup
            << ",\"repetitions\":" << config.repetitions
            << ",\"residency_capacity\":" << config.capacity
            << ",\"private_sessions_only\":" << (shared_substrate ? "false" : "true")
            << ",\"shared_resident_weights\":\""
            << (shared_substrate ? "qualification_mode" : "not_qualified") << "\""
            << ",\"kv_estimate_bytes\":" << kv_estimate
            << ",\"baseline_rss_kib\":" << baseline.rss_kib
            << ",\"peak_rss_kib\":" << peak.rss_kib
            << ",\"peak_rss_delta_kib\":" << (peak.rss_kib >= baseline.rss_kib ? peak.rss_kib - baseline.rss_kib : 0)
            << ",\"baseline_pss_kib\":" << baseline.pss_kib
            << ",\"peak_pss_kib\":" << peak.pss_kib
            << ",\"peak_pss_delta_kib\":" << (peak.pss_kib >= baseline.pss_kib ? peak.pss_kib - baseline.pss_kib : 0)
            << ",\"baseline_threads\":" << baseline.threads
            << ",\"peak_threads\":" << peak.threads
            << ",\"total_pair_time_ns\":" << total_ns
            << ",\"total_overlap_ns\":" << total_overlap_ns
            << ",\"valid\":" << (valid ? "true" : "false") << ",\"pairs\":[";
        for (size_t index = 0; index < pairs.size(); ++index) {
            if (index != 0) std::cout << ',';
            const PairRecord & pair = pairs[index];
            std::cout << "{\"pair_start_ns\":" << pair.start_ns
                << ",\"pair_end_ns\":" << pair.end_ns
                << ",\"makespan_ns\":" << pair.end_ns - pair.start_ns
                << ",\"overlap_ns\":" << pair.overlap_ns << ",\"a\":";
            print_record("a", pair.a);
            std::cout << ",\"b\":";
            print_record("b", pair.b);
            std::cout << '}';
        }
        std::cout << "]}\n";
        return valid ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "vbuf-step31y-backend-concurrency: " << error.what() << '\n';
        return 1;
    }
}

#pragma once

// Master-side and worker-side execution of the layer pipeline.
//
//   master : tokenise -> embed -> layers [0, a) -> ship activations
//   worker : layers [a, b) -> ship activations
//   last   : layers [b, L) -> output norm -> lm_head -> ship logits to master
//   master : sample, emit the token, repeat
//
// Every frame carries the sequence id it belongs to, so several requests can be
// in flight through the mesh at once: the master demultiplexes returning logits
// by sequence rather than assuming the next frame is its own.

#include "oracle/cluster_config.hpp"
#include "oracle/model/tokenizer.hpp"
#include "oracle/orch/scheduler.hpp"
#include "oracle/orch/worker_registry.hpp"
#include "oracle/runner/node_runner.hpp"
#include "oracle/security/gate.hpp"
#include "oracle/shard/memory_shard_manager.hpp"
#include "oracle/tb3/socket_transport.hpp"
#include "oracle/types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace oracle {

struct DagNode {
  NodeId id{0};
  LayerRange layers{};
  bool is_embed{false};
  bool is_lm_head{false};
};

struct HeartbeatState {
  NodeId id{0};
  std::chrono::steady_clock::time_point last_seen{};
  uint32_t misses{0};
  bool alive{true};
};

struct GenerateRequest {
  std::string model;
  std::string prompt;                          // used when `messages` is empty
  std::vector<model::ChatMessage> messages;    // rendered with the model's template
  uint32_t max_tokens{64};
  float temperature{0.0f};
  float top_p{1.0f};
  uint32_t top_k{0};
  float repeat_penalty{1.0f};
  uint64_t seed{0};
  std::vector<std::string> stop;
  bool stream{true};
  bool echo_prompt{false};
  uint64_t seq_id{0};
  int priority{0};
  std::string api_key_id;
  std::string request_id;
};

struct GenerateToken {
  uint32_t token_id{0};
  std::string text;
  bool stop{false};
  uint32_t index{0};
};

struct GenerateResult {
  uint32_t prompt_tokens{0};
  uint32_t completion_tokens{0};
  std::string finish_reason{"stop"};
  double prefill_ms{0};
  double decode_ms{0};
  double tokens_per_second{0};
  std::string text;
};

std::vector<DagNode> build_linear_dag(const ClusterConfig& cfg);

class PipelineOrchestrator {
 public:
  PipelineOrchestrator() = default;
  ~PipelineOrchestrator();

  Status init(ClusterConfig cfg, std::unique_ptr<NodeRunner> runner, NodeId self);
  Status configure_security(const security::SecurityConfig& cfg);
  Status configure_scheduler(const SchedulerConfig& cfg);

  Status start_transport();
  Status start_heartbeat();
  // Master: dial every worker, complete the signed handshake and start routing
  // returning logits.  Worker: dial the previous hop and register with it.
  Status connect_peers();
  void stop();

  // Runs one request to completion.  Executes on the calling thread so streaming
  // and backpressure follow the HTTP connection.
  Status generate(const GenerateRequest& req, const std::function<void(const GenerateToken&)>& on_token,
                  GenerateResult* result = nullptr);
  // Worker main loop: consume activations, run this node's layers, forward on.
  Status run_worker_loop();

  Status serve_openai(uint16_t port);

  [[nodiscard]] bool worker_alive(NodeId id) const;
  void on_heartbeat(NodeId id);
  void on_heartbeat(NodeId id, std::string_view payload);
  void tick_heartbeats();
  void publish_local_status();

  [[nodiscard]] const ClusterConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] MemoryShardManager& shards() { return *shards_; }
  TB3SocketTransport& transport() { return tx_; }
  NodeRunner* runner() { return runner_.get(); }
  WorkerRegistry& registry() { return registry_; }
  Scheduler& scheduler() { return scheduler_; }
  security::SecurityGate& security() { return security_; }
  [[nodiscard]] NodeId self() const noexcept { return self_; }
  [[nodiscard]] const std::vector<DagNode>& dag() const noexcept { return dag_; }
  [[nodiscard]] bool is_master() const;
  [[nodiscard]] bool single_node() const noexcept { return cfg_.nodes.size() <= 1; }
  [[nodiscard]] const std::string& model_name() const noexcept { return model_name_; }
  [[nodiscard]] const model::Tokenizer* tokenizer() const;
  // The GGUF backend, whichever runner wrapper is in front of it.
  [[nodiscard]] const class GgufRunner* gguf_backend() const;
  [[nodiscard]] uint64_t uptime_seconds() const;

  // ---- HTTP payloads (also used by the dashboard) ------------------------
  [[nodiscard]] std::string health_json() const;
  [[nodiscard]] std::string models_json() const;
  [[nodiscard]] std::string cluster_json(bool include_security) const;
  [[nodiscard]] std::string metrics_text() const;

  // Round-trips a small frame to each peer and records latency/throughput.
  Status benchmark_links(uint32_t payload_bytes, uint32_t iterations);

 private:
  struct SeqChannel {
    std::mutex mu;
    std::condition_variable cv;
    Tensor payload;
    bool ready{false};
    bool failed{false};
    std::string error;
  };

  Status build_dag();
  Status local_forward(const Tensor& in, KvCache& kv, Tensor* out, bool want_logits);
  Status run_stage(Tensor in, uint64_t seq_id, KvCache& kv, Tensor* logits, int timeout_ms);
  void router_loop();
  void reliability_loop();
  // Master side: bring every configured peer's activation link back up.
  void reconcile_links();
  void release_sequence(uint64_t seq_id);
  KvCache* kv_for(uint64_t seq_id, bool reset);
  void drop_sequence(uint64_t seq_id);
  [[nodiscard]] NodeId next_hop() const;
  [[nodiscard]] NodeId last_stage() const;
  [[nodiscard]] std::vector<int32_t> tokenize(const std::string& text, bool add_bos) const;
  [[nodiscard]] std::string detokenize(int32_t token) const;
  [[nodiscard]] std::string render_prompt(const GenerateRequest& req) const;
  uint32_t sample(std::span<const float> logits, const GenerateRequest& req,
                  const std::vector<int32_t>& history, uint64_t* rng_state) const;
  [[nodiscard]] ResourceRequirement requirement() const;

  ClusterConfig cfg_{};
  NodeId self_{0};
  std::string model_name_{"oracle"};
  std::unique_ptr<MemoryShardManager> shards_;
  std::unique_ptr<NodeRunner> runner_;
  TB3SocketTransport tx_;
  std::vector<DagNode> dag_;
  WorkerRegistry registry_;
  Scheduler scheduler_;
  security::SecurityGate security_;

  mutable std::mutex hb_mu_;
  std::unordered_map<NodeId, HeartbeatState> hb_;

  mutable std::mutex kv_mu_;
  std::unordered_map<uint64_t, std::unique_ptr<KvCache>> kv_;
  std::vector<uint64_t> kv_lru_;
  size_t max_sequences_{8};
  KvLayout kv_layout_{};

  std::mutex chan_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<SeqChannel>> channels_;

  std::unordered_map<NodeId, std::chrono::steady_clock::time_point> last_dial_;
  std::thread router_;
  std::thread reliability_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> requests_served_{0};
  std::atomic<uint64_t> tokens_generated_{0};
  std::chrono::steady_clock::time_point started_at_{};
};

Status run_openai_server(PipelineOrchestrator& orch, uint16_t port);
// The single-page dashboard served at "/".
const char* dashboard_html();

}  // namespace oracle

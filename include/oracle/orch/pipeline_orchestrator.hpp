#pragma once

#include "oracle/cluster_config.hpp"
#include "oracle/runner/node_runner.hpp"
#include "oracle/shard/memory_shard_manager.hpp"
#include "oracle/tb3/socket_transport.hpp"
#include "oracle/types.hpp"

#include <atomic>
#include <chrono>
#include <functional>
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
  std::string prompt;
  uint32_t max_tokens{64};
  float temperature{0.0f};
  bool stream{true};
  uint64_t seq_id{0};
};

struct GenerateToken {
  uint32_t token_id{0};
  std::string text;
  bool stop{false};
};

std::vector<DagNode> build_linear_dag(const ClusterConfig& cfg);

class PipelineOrchestrator {
 public:
  PipelineOrchestrator() = default;
  ~PipelineOrchestrator();

  Status init(ClusterConfig cfg, std::unique_ptr<NodeRunner> runner, NodeId self);
  Status start_transport();
  Status start_heartbeat();
  void stop();

  Status generate(const GenerateRequest& req, const std::function<void(const GenerateToken&)>& on_token);

  Status tensor_ring_once(Tensor t, Tensor* out, int timeout_ms);

  [[nodiscard]] bool worker_alive(NodeId id) const;
  void on_heartbeat(NodeId id);
  void tick_heartbeats();

  [[nodiscard]] const ClusterConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] MemoryShardManager& shards() { return *shards_; }
  TB3SocketTransport& transport() { return tx_; }
  NodeRunner* runner() { return runner_.get(); }
  [[nodiscard]] NodeId self() const noexcept { return self_; }
  [[nodiscard]] const std::vector<DagNode>& dag() const noexcept { return dag_; }

  Status serve_openai(uint16_t port);

 private:
  Status build_dag();
  Status local_forward(const Tensor& in, Tensor* out);
  uint32_t greedy_sample(std::span<const float> logits) const;
  std::vector<int32_t> tokenize(std::string_view text) const;
  std::string detokenize(uint32_t token) const;

  ClusterConfig cfg_{};
  NodeId self_{0};
  std::unique_ptr<MemoryShardManager> shards_;
  std::unique_ptr<NodeRunner> runner_;
  TB3SocketTransport tx_;
  std::vector<DagNode> dag_;
  KvCache kv_;
  mutable std::mutex hb_mu_;
  std::unordered_map<NodeId, HeartbeatState> hb_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> next_seq_{1};
};

Status run_openai_server(PipelineOrchestrator& orch, uint16_t port);

}  // namespace oracle

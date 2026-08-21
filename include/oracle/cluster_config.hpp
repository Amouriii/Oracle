#pragma once

#include "oracle/types.hpp"

#include <string>
#include <vector>

namespace oracle {

struct NodeConfig {
  NodeId id{0};
  std::string role{"worker"};  // master | worker
  std::string host{"127.0.0.1"};
  uint16_t http_port{8000};
  uint16_t transport_port{9200};
  uint16_t heartbeat_port{9100};
  double ram_budget_gb{32.0};
  double vram_budget_gb{4.0};
  LayerRange layers{};
};

// Policy knobs that live in the config file.  Kept as plain data here so the
// shard layer does not have to depend on the security library; the master maps
// them onto security::SecurityConfig at startup.
struct SecuritySettings {
  bool require_api_key{true};
  std::string api_key_file;
  std::string api_key_env{"ORACLE_API_KEYS"};
  std::string cluster_secret;
  std::string cluster_secret_env{"ORACLE_CLUSTER_SECRET"};
  bool require_worker_auth{true};
  uint64_t max_request_bytes{1ull << 20};
  uint32_t max_prompt_chars{131072};
  uint32_t max_completion_tokens{2048};
  uint32_t max_messages{256};
  uint32_t max_concurrent_requests{32};
  uint32_t max_concurrent_per_key{8};
  uint32_t requests_per_minute{120};
  uint32_t burst{20};
  uint32_t abuse_threshold{50};
  uint32_t ban_seconds{300};
  std::string audit_log_path;
  std::string model_manifest;
  bool verify_model_integrity{false};
  bool echo_security_log{false};
};

// Scheduler policy from the config file.
struct ServerSettings {
  uint32_t max_concurrent{4};
  uint32_t max_queue_depth{128};
  uint32_t queue_timeout_ms{30000};
  uint32_t request_timeout_ms{300000};
  uint32_t compute_threads{0};  // 0 = one per hardware thread
};

struct ClusterConfig {
  std::string name{"tb3-mesh"};
  uint16_t heartbeat_port{9100};
  uint16_t http_port{8000};
  uint16_t transport_port{9200};
  uint32_t mtu{9000};
  uint32_t heartbeat_misses{5};
  uint32_t heartbeat_interval_ms{200};
  // KV caches kept resident per node, i.e. how many sequences may be in flight
  // before the least recently used one is evicted and has to re-prefill.
  uint32_t max_sequences{8};
  ModelMeta model{};
  SecuritySettings security{};
  ServerSettings server{};
  std::vector<NodeConfig> nodes;

  [[nodiscard]] const NodeConfig* find(NodeId id) const;
  [[nodiscard]] const NodeConfig* master() const;
};

Status load_cluster_toml(const std::string& path, ClusterConfig* out);
Status save_cluster_toml(const std::string& path, const ClusterConfig& cfg);

}  // namespace oracle

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

struct ClusterConfig {
  std::string name{"tb3-mesh"};
  uint16_t heartbeat_port{9100};
  uint16_t http_port{8000};
  uint16_t transport_port{9200};
  uint32_t mtu{9000};
  uint32_t heartbeat_misses{5};
  uint32_t heartbeat_interval_ms{200};
  ModelMeta model{};
  std::vector<NodeConfig> nodes;

  [[nodiscard]] const NodeConfig* find(NodeId id) const;
  [[nodiscard]] const NodeConfig* master() const;
};

Status load_cluster_toml(const std::string& path, ClusterConfig* out);
Status save_cluster_toml(const std::string& path, const ClusterConfig& cfg);

}  // namespace oracle

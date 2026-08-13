#pragma once

#include "oracle/cluster_config.hpp"
#include "oracle/types.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace oracle {

struct MemorySnapshot {
  uint64_t total_bytes{0};
  uint64_t free_bytes{0};
  uint64_t used_bytes{0};
  uint64_t vram_recommended_bytes{0};
  uint64_t compressed_bytes{0};
  bool under_pressure{false};
};

class PressureMonitor {
 public:
  MemorySnapshot sample() const;
  Status refuse_if_overcommit(uint64_t extra_bytes, uint64_t ram_budget_bytes) const;
};

struct KvLayout {
  uint32_t n_local_layers{0};
  uint32_t n_kv_heads{0};
  uint32_t max_seq{0};
  uint32_t head_dim{0};
  DType dtype{DType::F16};
  uint64_t bytes_k{0};
  uint64_t bytes_v{0};
  uint64_t bytes_total{0};

  [[nodiscard]] uint64_t bytes_per_token() const noexcept {
    const uint64_t per_layer = 2ull * n_kv_heads * head_dim * dtype_size(dtype);
    return per_layer * n_local_layers;
  }
};

KvLayout plan_kv(const ModelMeta& model, LayerRange layers);

struct ShardPlan {
  ModelMeta model;
  std::vector<NodeConfig> nodes;
  uint64_t weight_bytes_total{0};
  uint64_t kv_bytes_total{0};
  bool feasible{false};
  std::string reason;
};

class MemoryShardManager {
 public:
  explicit MemoryShardManager(ClusterConfig cfg);

  Status plan(ShardPlan* out);
  Status assign_contiguous_layers();
  Status stream_layers_init(class TB3SocketTransport& tx, NodeId self,
                            const std::function<Status(NodeId, uint32_t layer, std::span<const std::byte>)>& send_blob);

  [[nodiscard]] const ClusterConfig& config() const noexcept { return cfg_; }
  ClusterConfig& config() noexcept { return cfg_; }
  [[nodiscard]] LayerRange layers_for(NodeId id) const;
  [[nodiscard]] const PressureMonitor& pressure() const noexcept { return pressure_; }

  static uint64_t estimate_layer_bytes(const ModelMeta& m);

 private:
  ClusterConfig cfg_;
  PressureMonitor pressure_;
};

}  // namespace oracle

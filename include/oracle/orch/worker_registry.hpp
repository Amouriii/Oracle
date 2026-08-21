#pragma once

// Live view of every node in the mesh: what it owns, what it has spare, and
// whether it is healthy enough to be given work.
//
// The registry is the reliability boundary.  A worker that misses its
// heartbeats is marked dead here, and the scheduler will not place a request on
// a stage whose only candidate is dead -- which is what stops a failed node from
// silently swallowing requests.

#include "oracle/cluster_config.hpp"
#include "oracle/types.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace oracle {

enum class WorkerState { Unknown, Joining, Ready, Busy, Degraded, Dead };

const char* worker_state_name(WorkerState s);

struct WorkerResources {
  NodeId id{0};
  std::string host;
  std::string role{"worker"};
  std::string runner;
  std::string model;
  LayerRange layers{};

  // Capacity, refreshed from the worker's heartbeat payload.
  uint32_t cpu_cores{0};
  double cpu_load{0.0};          // 0..1, normalised over cores
  uint64_t ram_total_bytes{0};
  uint64_t ram_free_bytes{0};
  uint64_t vram_total_bytes{0};
  uint64_t vram_free_bytes{0};
  uint64_t resident_weight_bytes{0};
  bool gpu_present{false};

  // Load and link quality.
  uint32_t active_requests{0};
  uint32_t queue_depth{0};
  uint32_t max_concurrent{4};
  double link_latency_ms{0.0};
  double link_gbps{0.0};

  WorkerState state{WorkerState::Unknown};
  uint32_t missed_heartbeats{0};
  uint64_t reconnects{0};
  std::chrono::steady_clock::time_point last_seen{};
  std::string last_error;

  [[nodiscard]] bool healthy() const noexcept {
    return state == WorkerState::Ready || state == WorkerState::Busy;
  }
  [[nodiscard]] bool accepting() const noexcept {
    return healthy() && active_requests < std::max(1u, max_concurrent);
  }
  // Serialised into the heartbeat datagram; parsed by note_heartbeat().
  [[nodiscard]] std::string encode_heartbeat() const;
  [[nodiscard]] std::string to_json() const;
};

// What one request needs from a stage before it can be placed there.
struct ResourceRequirement {
  std::string model;
  uint64_t ram_bytes{0};
  uint64_t kv_bytes{0};
  bool prefer_gpu{false};
};

struct RegistryConfig {
  uint32_t heartbeat_interval_ms{200};
  uint32_t heartbeat_misses{5};
  // A node that has been dead this long is retried rather than written off.
  uint32_t reconnect_after_ms{2000};
};

class WorkerRegistry {
 public:
  void configure(const RegistryConfig& cfg);
  // Seeds the registry from the static cluster config so the dashboard shows
  // every expected node, including ones that have not called in yet.
  void seed(const ClusterConfig& cfg, NodeId self);

  void upsert(const WorkerResources& w);
  void note_heartbeat(NodeId id, std::string_view payload);
  void note_join(NodeId id, const std::string& host, const std::string& runner, LayerRange layers);
  void mark_dead(NodeId id, const std::string& reason);
  void mark_state(NodeId id, WorkerState state);
  void note_reconnect(NodeId id);
  void add_active(NodeId id, int delta);
  void set_link(NodeId id, double latency_ms, double gbps);

  // Ages out nodes that have stopped sending heartbeats.  Returns the ids that
  // transitioned to Dead on this sweep, so the caller can log or react.
  std::vector<NodeId> tick();
  // Nodes that are dead but due for another connection attempt.
  [[nodiscard]] std::vector<NodeId> due_for_reconnect() const;

  [[nodiscard]] std::optional<WorkerResources> get(NodeId id) const;
  [[nodiscard]] std::vector<WorkerResources> snapshot() const;
  [[nodiscard]] size_t size() const;
  [[nodiscard]] size_t alive_count() const;
  [[nodiscard]] bool healthy(NodeId id) const;

  // Higher is better.  Combines spare RAM, CPU headroom, in-flight load and
  // link latency; returns a negative score for a node that cannot take the work.
  [[nodiscard]] double score(const WorkerResources& w, const ResourceRequirement& need) const;
  // Best healthy node covering exactly `layers`, or nullopt when every
  // candidate is dead or over capacity.
  [[nodiscard]] std::optional<WorkerResources> best_for_stage(LayerRange layers,
                                                              const ResourceRequirement& need) const;

  [[nodiscard]] std::string to_json() const;

 private:
  mutable std::mutex mu_;
  RegistryConfig cfg_{};
  std::unordered_map<NodeId, WorkerResources> workers_;
  std::unordered_map<NodeId, std::chrono::steady_clock::time_point> dead_since_;
  NodeId self_{0};
};

// Samples this process's own CPU/RAM so a worker can advertise itself.
WorkerResources sample_local_resources(NodeId id, const std::string& host, const std::string& role);

}  // namespace oracle

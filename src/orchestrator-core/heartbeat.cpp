#include "oracle/orch/pipeline_orchestrator.hpp"

#include <algorithm>
#include <chrono>

namespace oracle {

void PipelineOrchestrator::on_heartbeat(NodeId id) {
  std::lock_guard<std::mutex> g(hb_mu_);
  auto& s = hb_[id];
  s.id = id;
  s.last_seen = std::chrono::steady_clock::now();
  s.misses = 0;
  s.alive = true;
}

bool PipelineOrchestrator::worker_alive(NodeId id) const {
  std::lock_guard<std::mutex> g(hb_mu_);
  auto it = hb_.find(id);
  if (it == hb_.end()) {
    return false;
  }
  return it->second.alive;
}

void PipelineOrchestrator::tick_heartbeats() {
  const auto now = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::milliseconds(cfg_.heartbeat_interval_ms * std::max(1u, cfg_.heartbeat_misses));
  std::lock_guard<std::mutex> g(hb_mu_);
  for (auto& [id, s] : hb_) {
    if (id == self_) {
      s.alive = true;
      continue;
    }
    if (now - s.last_seen > timeout) {
      ++s.misses;
      if (s.misses >= cfg_.heartbeat_misses) {
        s.alive = false;
      }
    }
  }
}

}  // namespace oracle

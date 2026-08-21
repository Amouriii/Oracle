#include "oracle/orch/worker_registry.hpp"

#include "oracle/shard/memory_shard_manager.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace oracle {
namespace {

std::string json_escape(std::string_view s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
    }
    o += (static_cast<unsigned char>(c) < 0x20) ? ' ' : c;
  }
  return o;
}

// key=value pairs separated by ';' -- small, fits in one datagram, and needs no
// JSON parser on the receiving side.
std::unordered_map<std::string, std::string> parse_kv(std::string_view s) {
  std::unordered_map<std::string, std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    const auto end = std::min(s.find(';', i), s.size());
    const auto field = s.substr(i, end - i);
    const auto eq = field.find('=');
    if (eq != std::string_view::npos) {
      out.emplace(std::string(field.substr(0, eq)), std::string(field.substr(eq + 1)));
    }
    i = end + 1;
  }
  return out;
}

double to_double(const std::unordered_map<std::string, std::string>& m, const char* k, double def) {
  const auto it = m.find(k);
  if (it == m.end()) {
    return def;
  }
  try {
    return std::stod(it->second);
  } catch (...) {
    return def;
  }
}

uint64_t to_u64(const std::unordered_map<std::string, std::string>& m, const char* k, uint64_t def) {
  const auto it = m.find(k);
  if (it == m.end()) {
    return def;
  }
  try {
    return std::stoull(it->second);
  } catch (...) {
    return def;
  }
}

std::string to_str(const std::unordered_map<std::string, std::string>& m, const char* k,
                   const std::string& def) {
  const auto it = m.find(k);
  return it == m.end() ? def : it->second;
}

}  // namespace

const char* worker_state_name(WorkerState s) {
  switch (s) {
    case WorkerState::Unknown: return "unknown";
    case WorkerState::Joining: return "joining";
    case WorkerState::Ready: return "ready";
    case WorkerState::Busy: return "busy";
    case WorkerState::Degraded: return "degraded";
    case WorkerState::Dead: return "dead";
  }
  return "unknown";
}

std::string WorkerResources::encode_heartbeat() const {
  std::ostringstream os;
  os << "cores=" << cpu_cores << ";load=" << cpu_load << ";ram_total=" << ram_total_bytes
     << ";ram_free=" << ram_free_bytes << ";vram_total=" << vram_total_bytes
     << ";vram_free=" << vram_free_bytes << ";resident=" << resident_weight_bytes
     << ";active=" << active_requests << ";queued=" << queue_depth << ";maxc=" << max_concurrent
     << ";gpu=" << (gpu_present ? 1 : 0) << ";l0=" << layers.start << ";l1=" << layers.end
     << ";runner=" << runner << ";model=" << model;
  return os.str();
}

std::string WorkerResources::to_json() const {
  const auto now = std::chrono::steady_clock::now();
  const double age_ms =
      last_seen.time_since_epoch().count()
          ? std::chrono::duration<double, std::milli>(now - last_seen).count()
          : -1.0;
  std::ostringstream os;
  os << "{\"id\":" << id << ",\"host\":\"" << json_escape(host) << "\",\"role\":\"" << json_escape(role)
     << "\",\"runner\":\"" << json_escape(runner) << "\",\"model\":\"" << json_escape(model)
     << "\",\"state\":\"" << worker_state_name(state) << "\""
     << ",\"layers\":{\"start\":" << layers.start << ",\"end\":" << layers.end
     << ",\"count\":" << layers.count() << "}"
     << ",\"cpu\":{\"cores\":" << cpu_cores << ",\"load\":" << cpu_load << "}"
     << ",\"ram\":{\"total\":" << ram_total_bytes << ",\"free\":" << ram_free_bytes
     << ",\"resident_weights\":" << resident_weight_bytes << "}"
     << ",\"gpu\":{\"present\":" << (gpu_present ? "true" : "false") << ",\"total\":" << vram_total_bytes
     << ",\"free\":" << vram_free_bytes << "}"
     << ",\"load\":{\"active\":" << active_requests << ",\"queued\":" << queue_depth
     << ",\"max_concurrent\":" << max_concurrent << "}"
     << ",\"link\":{\"latency_ms\":" << link_latency_ms << ",\"gbps\":" << link_gbps << "}"
     << ",\"heartbeat\":{\"age_ms\":" << age_ms << ",\"missed\":" << missed_heartbeats
     << ",\"reconnects\":" << reconnects << "}"
     << ",\"link_up\":" << (link_up ? "true" : "false")
     << ",\"accepting\":" << (accepting() ? "true" : "false");
  if (!last_error.empty()) {
    os << ",\"last_error\":\"" << json_escape(last_error) << "\"";
  }
  os << "}";
  return os.str();
}

void WorkerRegistry::configure(const RegistryConfig& cfg) {
  std::lock_guard<std::mutex> g(mu_);
  cfg_ = cfg;
}

void WorkerRegistry::seed(const ClusterConfig& cfg, NodeId self) {
  std::lock_guard<std::mutex> g(mu_);
  self_ = self;
  cfg_.heartbeat_interval_ms = cfg.heartbeat_interval_ms;
  cfg_.heartbeat_misses = cfg.heartbeat_misses;
  for (const auto& n : cfg.nodes) {
    auto& w = workers_[n.id];
    w.id = n.id;
    w.host = n.host;
    w.role = n.role;
    w.layers = n.layers;
    w.model = cfg.model.name;
    if (w.ram_total_bytes == 0) {
      w.ram_total_bytes = static_cast<uint64_t>(n.ram_budget_gb * (1ull << 30));
      w.ram_free_bytes = w.ram_total_bytes;
    }
    if (w.vram_total_bytes == 0) {
      w.vram_total_bytes = static_cast<uint64_t>(n.vram_budget_gb * (1ull << 30));
      w.vram_free_bytes = w.vram_total_bytes;
      w.gpu_present = w.vram_total_bytes > 0;
    }
    if (n.id == self) {
      w.state = WorkerState::Ready;
      w.link_up = true;
      w.last_seen = std::chrono::steady_clock::now();
    } else if (w.state == WorkerState::Unknown) {
      w.state = WorkerState::Joining;
      w.link_up = false;  // nothing is connected until the handshake completes
    }
  }
}

void WorkerRegistry::upsert(const WorkerResources& w) {
  std::lock_guard<std::mutex> g(mu_);
  auto& dst = workers_[w.id];
  const auto reconnects = dst.reconnects;
  dst = w;
  dst.reconnects = std::max(reconnects, w.reconnects);
}

void WorkerRegistry::note_heartbeat(NodeId id, std::string_view payload) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  w.last_seen = std::chrono::steady_clock::now();
  w.missed_heartbeats = 0;
  // A heartbeat proves the process is alive, so it clears Degraded and Dead
  // alike -- a node that missed a beat and then came back must not stay
  // Degraded forever.  It says nothing about the activation stream, though:
  // `link_up` is owned by the transport reconciliation loop.
  if (w.state != WorkerState::Busy) {
    w.state = WorkerState::Ready;
  }
  dead_since_.erase(id);
  if (payload.empty()) {
    return;
  }
  const auto kv = parse_kv(payload);
  w.cpu_cores = static_cast<uint32_t>(to_u64(kv, "cores", w.cpu_cores));
  w.cpu_load = to_double(kv, "load", w.cpu_load);
  w.ram_total_bytes = to_u64(kv, "ram_total", w.ram_total_bytes);
  w.ram_free_bytes = to_u64(kv, "ram_free", w.ram_free_bytes);
  w.vram_total_bytes = to_u64(kv, "vram_total", w.vram_total_bytes);
  w.vram_free_bytes = to_u64(kv, "vram_free", w.vram_free_bytes);
  w.resident_weight_bytes = to_u64(kv, "resident", w.resident_weight_bytes);
  w.active_requests = static_cast<uint32_t>(to_u64(kv, "active", w.active_requests));
  w.queue_depth = static_cast<uint32_t>(to_u64(kv, "queued", w.queue_depth));
  w.max_concurrent = static_cast<uint32_t>(to_u64(kv, "maxc", w.max_concurrent));
  w.gpu_present = to_u64(kv, "gpu", w.gpu_present ? 1 : 0) != 0;
  w.layers.start = static_cast<uint32_t>(to_u64(kv, "l0", w.layers.start));
  w.layers.end = static_cast<uint32_t>(to_u64(kv, "l1", w.layers.end));
  w.runner = to_str(kv, "runner", w.runner);
  w.model = to_str(kv, "model", w.model);
  if (w.active_requests > 0 && w.state == WorkerState::Ready) {
    w.state = WorkerState::Busy;
  } else if (w.active_requests == 0 && w.state == WorkerState::Busy) {
    w.state = WorkerState::Ready;
  }
}

void WorkerRegistry::note_join(NodeId id, const std::string& host, const std::string& runner,
                               LayerRange layers) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  if (!host.empty()) {
    w.host = host;
  }
  if (!runner.empty()) {
    w.runner = runner;
  }
  if (layers.end > layers.start) {
    w.layers = layers;
  }
  w.state = WorkerState::Ready;
  w.link_up = true;
  w.missed_heartbeats = 0;
  w.last_seen = std::chrono::steady_clock::now();
  w.last_error.clear();
  dead_since_.erase(id);
}

void WorkerRegistry::mark_dead(NodeId id, const std::string& reason) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  w.state = WorkerState::Dead;
  w.link_up = false;
  w.last_error = reason;
  w.active_requests = 0;
  dead_since_[id] = std::chrono::steady_clock::now();
}

void WorkerRegistry::mark_state(NodeId id, WorkerState state) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  w.state = state;
  if (state != WorkerState::Dead) {
    dead_since_.erase(id);
  }
}

void WorkerRegistry::note_reconnect(NodeId id) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  ++w.reconnects;
  w.state = WorkerState::Ready;
  w.link_up = true;
  w.missed_heartbeats = 0;
  w.last_seen = std::chrono::steady_clock::now();
  dead_since_.erase(id);
}

void WorkerRegistry::set_link_up(NodeId id, bool up) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  if (w.link_up == up) {
    return;
  }
  w.link_up = up;
  if (!up) {
    w.active_requests = 0;
    if (w.last_error.empty()) {
      w.last_error = "activation link is down";
    }
  } else {
    w.last_error.clear();
  }
}

void WorkerRegistry::add_active(NodeId id, int delta) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  if (delta >= 0) {
    w.active_requests += static_cast<uint32_t>(delta);
  } else {
    const auto sub = static_cast<uint32_t>(-delta);
    w.active_requests = w.active_requests > sub ? w.active_requests - sub : 0;
  }
  if (w.state == WorkerState::Ready && w.active_requests > 0) {
    w.state = WorkerState::Busy;
  } else if (w.state == WorkerState::Busy && w.active_requests == 0) {
    w.state = WorkerState::Ready;
  }
}

void WorkerRegistry::set_link(NodeId id, double latency_ms, double gbps) {
  std::lock_guard<std::mutex> g(mu_);
  auto& w = workers_[id];
  w.id = id;
  if (latency_ms > 0) {
    // Exponential smoothing: one slow frame should nudge the estimate, not
    // rewrite it, or the scheduler will chase noise.
    w.link_latency_ms = w.link_latency_ms > 0 ? 0.8 * w.link_latency_ms + 0.2 * latency_ms : latency_ms;
  }
  if (gbps > 0) {
    w.link_gbps = w.link_gbps > 0 ? 0.8 * w.link_gbps + 0.2 * gbps : gbps;
  }
}

std::vector<NodeId> WorkerRegistry::tick() {
  const auto now = std::chrono::steady_clock::now();
  std::vector<NodeId> newly_dead;
  std::lock_guard<std::mutex> g(mu_);
  const auto window = std::chrono::milliseconds(std::max(50u, cfg_.heartbeat_interval_ms) *
                                                std::max(1u, cfg_.heartbeat_misses));
  for (auto& [id, w] : workers_) {
    if (id == self_ || w.state == WorkerState::Dead) {
      continue;
    }
    if (w.last_seen.time_since_epoch().count() == 0) {
      continue;  // never seen; still Joining
    }
    if (now - w.last_seen > window) {
      ++w.missed_heartbeats;
      if (w.missed_heartbeats >= cfg_.heartbeat_misses) {
        w.state = WorkerState::Dead;
        w.active_requests = 0;
        w.last_error = "missed " + std::to_string(w.missed_heartbeats) + " heartbeats";
        dead_since_[id] = now;
        newly_dead.push_back(id);
      } else {
        w.state = WorkerState::Degraded;
      }
    }
  }
  return newly_dead;
}

std::vector<NodeId> WorkerRegistry::due_for_reconnect() const {
  const auto now = std::chrono::steady_clock::now();
  std::vector<NodeId> out;
  std::lock_guard<std::mutex> g(mu_);
  for (const auto& [id, since] : dead_since_) {
    if (now - since >= std::chrono::milliseconds(cfg_.reconnect_after_ms)) {
      out.push_back(id);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::optional<WorkerResources> WorkerRegistry::get(NodeId id) const {
  std::lock_guard<std::mutex> g(mu_);
  const auto it = workers_.find(id);
  return it == workers_.end() ? std::nullopt : std::optional<WorkerResources>(it->second);
}

std::vector<WorkerResources> WorkerRegistry::snapshot() const {
  std::lock_guard<std::mutex> g(mu_);
  std::vector<WorkerResources> out;
  out.reserve(workers_.size());
  for (const auto& [id, w] : workers_) {
    out.push_back(w);
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  return out;
}

size_t WorkerRegistry::size() const {
  std::lock_guard<std::mutex> g(mu_);
  return workers_.size();
}

size_t WorkerRegistry::alive_count() const {
  std::lock_guard<std::mutex> g(mu_);
  size_t n = 0;
  for (const auto& [id, w] : workers_) {
    if (w.healthy()) {
      ++n;
    }
  }
  return n;
}

bool WorkerRegistry::healthy(NodeId id) const {
  std::lock_guard<std::mutex> g(mu_);
  const auto it = workers_.find(id);
  return it != workers_.end() && it->second.healthy();
}

double WorkerRegistry::score(const WorkerResources& w, const ResourceRequirement& need) const {
  if (!w.accepting()) {
    return -1.0;
  }
  if (!need.model.empty() && !w.model.empty() && w.model != need.model) {
    return -1.0;
  }
  const uint64_t required = need.ram_bytes + need.kv_bytes;
  if (required && w.ram_free_bytes && w.ram_free_bytes < required) {
    return -1.0;
  }

  // Weighted blend: free memory and idle capacity dominate, link quality breaks
  // ties, and in-flight work is penalised so load spreads rather than piles up.
  const double ram_headroom =
      w.ram_total_bytes ? static_cast<double>(w.ram_free_bytes) / static_cast<double>(w.ram_total_bytes)
                        : 0.5;
  const double cpu_headroom = std::clamp(1.0 - w.cpu_load, 0.0, 1.0);
  const double slots = std::max(1u, w.max_concurrent);
  const double load_headroom = std::clamp(1.0 - w.active_requests / slots, 0.0, 1.0);
  const double queue_penalty = 1.0 / (1.0 + static_cast<double>(w.queue_depth));
  const double latency_term = 1.0 / (1.0 + std::max(0.0, w.link_latency_ms));
  const double gpu_bonus = (need.prefer_gpu && w.gpu_present && w.vram_free_bytes) ? 0.15 : 0.0;

  return 0.30 * ram_headroom + 0.20 * cpu_headroom + 0.25 * load_headroom + 0.10 * queue_penalty +
         0.15 * latency_term + gpu_bonus;
}

std::optional<WorkerResources> WorkerRegistry::best_for_stage(LayerRange layers,
                                                              const ResourceRequirement& need) const {
  const auto all = snapshot();
  std::optional<WorkerResources> best;
  double best_score = 0.0;
  for (const auto& w : all) {
    if (w.layers.start != layers.start || w.layers.end != layers.end) {
      continue;
    }
    const double s = score(w, need);
    if (s < 0) {
      continue;
    }
    if (!best || s > best_score) {
      best = w;
      best_score = s;
    }
  }
  return best;
}

std::string WorkerRegistry::to_json() const {
  const auto all = snapshot();
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < all.size(); ++i) {
    os << (i ? "," : "") << all[i].to_json();
  }
  os << "]";
  return os.str();
}

WorkerResources sample_local_resources(NodeId id, const std::string& host, const std::string& role) {
  WorkerResources w;
  w.id = id;
  w.host = host;
  w.role = role;
  w.state = WorkerState::Ready;
  w.last_seen = std::chrono::steady_clock::now();
  const unsigned hw = std::thread::hardware_concurrency();
  w.cpu_cores = hw ? hw : 1;

  PressureMonitor pm;
  const auto snap = pm.sample();
  w.ram_total_bytes = snap.total_bytes;
  w.ram_free_bytes = snap.free_bytes;

#if !defined(__APPLE__)
  // /proc/loadavg is the cheapest portable proxy for "how busy is this box".
  std::ifstream la("/proc/loadavg");
  double one_min = 0.0;
  if (la >> one_min && w.cpu_cores) {
    w.cpu_load = std::clamp(one_min / static_cast<double>(w.cpu_cores), 0.0, 1.0);
  }
#endif
  return w;
}

}  // namespace oracle

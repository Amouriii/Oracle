#include "oracle/shard/memory_shard_manager.hpp"

#include "check.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / "oracle-cluster-test.toml";
  {
    std::ofstream out(path);
    out << R"(
[cluster]
name = "test"
heartbeat_port = 9100
http_port = 8000
transport_port = 9200

[model]
name = "llama-70b-q4"
n_layers = 80
hidden_dim = 8192
n_heads = 64
n_kv_heads = 8
head_dim = 128
n_vocab = 32000
max_seq = 4096
bytes_per_layer = 500000000

[[nodes]]
id = 0
role = "master"
host = "10.10.0.1"
ram_budget_gb = 32
vram_budget_gb = 4

[[nodes]]
id = 1
role = "worker"
host = "10.10.0.2"
ram_budget_gb = 32
vram_budget_gb = 4

[[nodes]]
id = 2
role = "worker"
host = "10.10.0.3"
ram_budget_gb = 32
vram_budget_gb = 4
)";
  }
  oracle::ClusterConfig cfg;
  auto st = oracle::load_cluster_toml(path.string(), &cfg);
  if (!st) {
    std::cerr << st.message << "\n";
    return 1;
  }
  CHECK(cfg.nodes.size() == 3);
  CHECK(cfg.model.n_layers == 80);

  oracle::MemoryShardManager mgr(cfg);
  oracle::ShardPlan plan;
  st = mgr.plan(&plan);
  if (!st) {
    std::cerr << st.message << "\n";
    return 1;
  }
  CHECK(plan.nodes[0].layers.start == 0);
  CHECK(plan.nodes[2].layers.end == 80);
  uint32_t covered = 0;
  for (const auto& n : plan.nodes) {
    covered += n.layers.count();
    const auto kv = oracle::plan_kv(cfg.model, n.layers);
    CHECK(kv.n_local_layers == n.layers.count());
    CHECK(kv.bytes_total > 0);
  }
  CHECK(covered == 80);
  std::cout << "test_shard_plan ok feasible=" << plan.feasible << " kv_bytes=" << plan.kv_bytes_total
            << " reason=" << plan.reason << "\n";

  oracle::PressureMonitor pm;
  const auto snap = pm.sample();
  std::cout << "host total_gb=" << (snap.total_bytes / (1ull << 30)) << " free_gb="
            << (snap.free_bytes / (1ull << 30)) << "\n";
  return 0;
}

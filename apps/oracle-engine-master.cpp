#include "oracle/cluster_config.hpp"
#include "oracle/orch/pipeline_orchestrator.hpp"

#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string cfg_path = "configs/cluster.toml";
  std::string runner = "accelerate";
  oracle::NodeId id = 0;
  bool single = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      cfg_path = argv[++i];
    } else if (std::strcmp(argv[i], "--id") == 0 && i + 1 < argc) {
      id = static_cast<oracle::NodeId>(std::stoul(argv[++i]));
    } else if (std::strcmp(argv[i], "--runner") == 0 && i + 1 < argc) {
      runner = argv[++i];
    } else if (std::strcmp(argv[i], "--single") == 0) {
      single = true;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cout << "oracle-engine-master --config PATH [--id N] [--runner accelerate|metal|llamacpp] [--single]\n";
      return 0;
    }
  }
  oracle::ClusterConfig cfg;
  auto st = oracle::load_cluster_toml(cfg_path, &cfg);
  if (!st) {
    std::cerr << "config: " << st.message << "\n";
    return 1;
  }
  if (single && !cfg.nodes.empty()) {
    cfg.nodes.resize(1);
    cfg.nodes[0].role = "master";
    cfg.nodes[0].host = "127.0.0.1";
  }
  oracle::PipelineOrchestrator orch;
  st = orch.init(cfg, oracle::make_runner(runner), id);
  if (!st) {
    std::cerr << "init: " << st.message << "\n";
    return 1;
  }
  if (!single) {
    st = orch.start_transport();
    if (!st) {
      std::cerr << "transport: " << st.message << "\n";
      return 1;
    }
    st = orch.start_heartbeat();
    if (!st) {
      std::cerr << "heartbeat: " << st.message << "\n";
    }
  }
  const uint16_t port = cfg.http_port ? cfg.http_port : 8000;
  std::cout << "oracle-engine-master listening OpenAI http://0.0.0.0:" << port << " runner=" << runner
            << " single=" << single << "\n";
  st = orch.serve_openai(port);
  if (!st) {
    std::cerr << "http: " << st.message << "\n";
    return 1;
  }
  return 0;
}

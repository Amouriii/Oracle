// Oracle worker: owns one contiguous slice of the model's layers, receives
// activations from the previous hop and forwards them to the next.
#include "oracle/cluster_config.hpp"
#include "oracle/compute/blas.hpp"
#include "oracle/orch/pipeline_orchestrator.hpp"

#include <csignal>
#include <cstring>
#include <iostream>
#include <string>

namespace {

oracle::PipelineOrchestrator* g_orch = nullptr;

void on_signal(int) {
  if (g_orch) {
    g_orch->stop();
  }
  std::_Exit(0);
}

void usage() {
  std::cout <<
      "oracle-engine-worker --id N [options]\n"
      "  --config PATH   cluster TOML (default configs/cluster.toml)\n"
      "  --id N          this node's id (required)\n"
      "  --model PATH    GGUF file, overriding [model] path\n"
      "  --runner KIND   auto | gguf | llamacpp | accelerate | metal\n"
      "  --threads N     compute threads\n"
      "  --help\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string cfg_path = "configs/cluster.toml";
  std::string runner = "auto";
  std::string model_override;
  oracle::NodeId id = 1;
  bool have_id = false;
  int threads = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << a << " needs " << what << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--config") {
      cfg_path = next("a path");
    } else if (a == "--id") {
      id = static_cast<oracle::NodeId>(std::stoul(next("a node id")));
      have_id = true;
    } else if (a == "--model") {
      model_override = next("a path");
    } else if (a == "--runner") {
      runner = next("a runner kind");
    } else if (a == "--threads") {
      threads = std::stoi(next("a thread count"));
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::cerr << "unknown option " << a << "\n";
      usage();
      return 2;
    }
  }
  if (!have_id) {
    std::cerr << "--id is required: a worker must know which layer shard it owns\n";
    return 2;
  }

  oracle::ClusterConfig cfg;
  auto st = oracle::load_cluster_toml(cfg_path, &cfg);
  if (!st) {
    std::cerr << "config: " << st.message << "\n";
    return 1;
  }
  if (!model_override.empty()) {
    cfg.model.path = model_override;
  }
  if (threads > 0) {
    oracle::compute::set_thread_count(threads);
  } else if (cfg.server.compute_threads > 0) {
    oracle::compute::set_thread_count(static_cast<int>(cfg.server.compute_threads));
  }

  oracle::PipelineOrchestrator orch;
  g_orch = &orch;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGPIPE, SIG_IGN);

  st = orch.init(cfg, oracle::make_runner(runner), id);
  if (!st) {
    std::cerr << "init: " << st.message << "\n";
    return 1;
  }

  // A worker enforces the same cluster secret as the master; it has no HTTP
  // surface, so API keys are irrelevant here.
  oracle::security::SecurityConfig sec;
  sec.require_api_key = false;
  sec.api_key_env.clear();
  sec.require_worker_auth = cfg.security.require_worker_auth;
  sec.cluster_secret = cfg.security.cluster_secret;
  sec.cluster_secret_env = cfg.security.cluster_secret_env;
  sec.audit_log_path = cfg.security.audit_log_path;
  sec.model_manifest_path = cfg.security.model_manifest;
  sec.verify_model_integrity = cfg.security.verify_model_integrity;
  sec.echo_security_log = cfg.security.echo_security_log;
  st = orch.configure_security(sec);
  if (!st) {
    std::cerr << "security: " << st.message << "\n";
    return 1;
  }

  st = orch.start_transport();
  if (!st) {
    std::cerr << "listen: " << st.message << "\n";
    return 1;
  }
  st = orch.start_heartbeat();
  if (!st) {
    std::cerr << "heartbeat: " << st.message << "\n";
    return 1;
  }

  const auto* me = orch.config().find(id);
  std::cout << "Oracle worker\n";
  std::cout << "  node     " << id << " at " << (me ? me->host : "?") << ":"
            << (me ? me->transport_port : 0) << "\n";
  std::cout << "  layers   [" << (me ? me->layers.start : 0) << ", " << (me ? me->layers.end : 0)
            << ")\n";
  std::cout << "  runner   " << (orch.runner() ? orch.runner()->name() : "none") << " / "
            << oracle::compute::backend_name() << " x" << oracle::compute::thread_count()
            << " threads\n";
  std::cout << "  waiting for the master to register...\n";

  st = orch.connect_peers();
  if (!st) {
    std::cerr << "cluster: " << st.message << "\n";
    return 1;
  }
  std::cout << "  joined; serving activations\n";

  st = orch.run_worker_loop();
  if (!st) {
    std::cerr << "worker loop: " << st.message << "\n";
    return 1;
  }
  return 0;
}

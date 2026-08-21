// Oracle master: owns the first pipeline stage, the OpenAI HTTP surface, the
// request scheduler and the security gate.
#include "oracle/cluster_config.hpp"
#include "oracle/compute/blas.hpp"
#include "oracle/model/gguf.hpp"
#include "oracle/orch/pipeline_orchestrator.hpp"
#include "oracle/security/api_keys.hpp"

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
      "oracle-engine-master [options]\n"
      "  --config PATH        cluster TOML (default configs/cluster.toml)\n"
      "  --id N               this node's id (default: the config's master)\n"
      "  --model PATH         GGUF file, overriding [model] path\n"
      "  --runner KIND        auto | gguf | llamacpp | accelerate | metal\n"
      "  --port N             HTTP port, overriding the config\n"
      "  --threads N          compute threads (default: one per hardware thread)\n"
      "  --single             run the whole model on this node\n"
      "  --no-auth            disable API-key authentication (development only)\n"
      "  --api-key SECRET     install a single API key named 'cli'\n"
      "  --generate-key       print a fresh random API key and exit\n"
      "  --help\n";
}

oracle::security::SecurityConfig to_security(const oracle::SecuritySettings& s) {
  oracle::security::SecurityConfig c;
  c.require_api_key = s.require_api_key;
  c.api_key_file = s.api_key_file;
  c.api_key_env = s.api_key_env;
  c.cluster_secret = s.cluster_secret;
  c.cluster_secret_env = s.cluster_secret_env;
  c.require_worker_auth = s.require_worker_auth;
  c.max_request_bytes = s.max_request_bytes;
  c.max_prompt_chars = s.max_prompt_chars;
  c.max_completion_tokens = s.max_completion_tokens;
  c.max_messages = s.max_messages;
  c.max_concurrent_requests = s.max_concurrent_requests;
  c.max_concurrent_per_key = s.max_concurrent_per_key;
  c.rate.requests_per_minute = s.requests_per_minute;
  c.rate.burst = s.burst;
  c.rate.abuse_threshold = s.abuse_threshold;
  c.rate.ban_seconds = s.ban_seconds;
  c.audit_log_path = s.audit_log_path;
  c.model_manifest_path = s.model_manifest;
  c.verify_model_integrity = s.verify_model_integrity;
  c.echo_security_log = s.echo_security_log;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  std::string cfg_path = "configs/cluster.toml";
  std::string runner = "auto";
  std::string model_override;
  std::string cli_key;
  oracle::NodeId id = 0;
  bool have_id = false;
  bool single = false;
  bool no_auth = false;
  int port_override = 0;
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
    } else if (a == "--port") {
      port_override = std::stoi(next("a port"));
    } else if (a == "--threads") {
      threads = std::stoi(next("a thread count"));
    } else if (a == "--single") {
      single = true;
    } else if (a == "--no-auth") {
      no_auth = true;
    } else if (a == "--api-key") {
      cli_key = next("a secret");
    } else if (a == "--generate-key") {
      const auto secret = oracle::security::ApiKeyStore::generate_secret();
      if (secret.empty()) {
        std::cerr << "could not read /dev/urandom; refusing to print a weak key\n";
        return 1;
      }
      std::cout << secret << "\n";
      return 0;
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::cerr << "unknown option " << a << "\n";
      usage();
      return 2;
    }
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
  if (single && !cfg.nodes.empty()) {
    cfg.nodes.resize(1);
    cfg.nodes[0].role = "master";
    cfg.nodes[0].host = "127.0.0.1";
    id = cfg.nodes[0].id;
    have_id = true;
  }
  if (!have_id) {
    const auto* master = cfg.master();
    id = master ? master->id : 0;
  }
  if (no_auth) {
    cfg.security.require_api_key = false;
  }
  if (single) {
    // A single node has no peers to authenticate.
    cfg.security.require_worker_auth = false;
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

  auto sec = to_security(cfg.security);
  if (!cli_key.empty()) {
    auto add = orch.security().keys().add_plaintext("cli", cli_key, true);
    if (!add) {
      std::cerr << "api key: " << add.message << "\n";
      return 1;
    }
  }
  st = orch.configure_security(sec);
  if (!st) {
    std::cerr << "security: " << st.message << "\n";
    return 1;
  }

  oracle::SchedulerConfig sched;
  sched.max_concurrent = cfg.server.max_concurrent;
  sched.max_queue_depth = cfg.server.max_queue_depth;
  sched.queue_timeout_ms = cfg.server.queue_timeout_ms;
  sched.request_timeout_ms = cfg.server.request_timeout_ms;
  (void)orch.configure_scheduler(sched);

  if (!single) {
    st = orch.start_transport();
    if (!st) {
      std::cerr << "transport: " << st.message << "\n";
      return 1;
    }
    st = orch.start_heartbeat();
    if (!st) {
      std::cerr << "heartbeat: " << st.message << "\n";
      return 1;
    }
    st = orch.connect_peers();
    if (!st) {
      std::cerr << "cluster: " << st.message << "\n";
      return 1;
    }
    (void)orch.benchmark_links(65536, 8);
  }

  const auto& live = orch.config();
  const auto* me = live.find(id);
  const uint16_t http_port = port_override ? static_cast<uint16_t>(port_override)
                                           : (me && me->http_port ? me->http_port : live.http_port);

  std::cout << "Oracle master\n";
  std::cout << "  node        " << id << " (" << (me ? me->role : "master") << ")\n";
  std::cout << "  model       " << orch.model_name();
  if (!live.model.path.empty()) {
    std::cout << "  <- " << live.model.path;
  }
  std::cout << "\n";
  std::cout << "  layers      " << live.model.n_layers << " across " << live.nodes.size() << " node(s)\n";
  std::cout << "  runner      " << (orch.runner() ? orch.runner()->name() : "none") << " / "
            << oracle::compute::backend_name() << " x" << oracle::compute::thread_count() << " threads\n";
  std::cout << "  auth        " << (cfg.security.require_api_key ? "required" : "DISABLED") << "\n";
  std::cout << "  http        http://0.0.0.0:" << http_port << "  (dashboard at /)\n";

  st = orch.serve_openai(http_port);
  if (!st) {
    std::cerr << "http: " << st.message << "\n";
    return 1;
  }
  return 0;
}

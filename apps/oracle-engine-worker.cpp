#include "oracle/cluster_config.hpp"
#include "oracle/orch/pipeline_orchestrator.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
  std::string cfg_path = "configs/cluster.toml";
  std::string runner = "accelerate";
  oracle::NodeId id = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      cfg_path = argv[++i];
    } else if (std::strcmp(argv[i], "--id") == 0 && i + 1 < argc) {
      id = static_cast<oracle::NodeId>(std::stoul(argv[++i]));
    } else if (std::strcmp(argv[i], "--runner") == 0 && i + 1 < argc) {
      runner = argv[++i];
    }
  }

  oracle::ClusterConfig cfg;
  auto st = oracle::load_cluster_toml(cfg_path, &cfg);
  if (!st) {
    std::cerr << "config: " << st.message << "\n";
    return 1;
  }

  oracle::PipelineOrchestrator orch;
  st = orch.init(cfg, oracle::make_runner(runner), id);
  if (!st) {
    std::cerr << "init: " << st.message << "\n";
    return 1;
  }
  st = orch.start_transport();
  if (!st) {
    std::cerr << "listen: " << st.message << "\n";
    return 1;
  }
  (void)orch.start_heartbeat();

  oracle::NodeId prev = 0;
  oracle::NodeId next = 0;
  bool last = false;
  const oracle::NodeConfig* prev_node = nullptr;
  const oracle::NodeConfig* master = cfg.master();
  for (size_t i = 0; i < cfg.nodes.size(); ++i) {
    if (cfg.nodes[i].id != id) {
      continue;
    }
    last = (i + 1 == cfg.nodes.size());
    if (i > 0) {
      prev = cfg.nodes[i - 1].id;
      prev_node = &cfg.nodes[i - 1];
    }
    if (!last) {
      next = cfg.nodes[i + 1].id;
    }
  }

  if (prev_node) {
    bool ok = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
      auto cs = orch.transport().connect(prev, prev_node->host, prev_node->transport_port);
      if (cs) {
        ok = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (!ok) {
      std::cerr << "failed to connect to previous hop " << prev_node->host << "\n";
      return 1;
    }
  }
  if (last && master) {
    for (int attempt = 0; attempt < 50; ++attempt) {
      auto cs = orch.transport().connect(master->id, master->host, master->transport_port);
      if (cs) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }

  oracle::KvCache kv;
  kv.allocate(oracle::plan_kv(orch.config().model, orch.runner()->layers()));
  std::cout << "oracle-engine-worker id=" << id << " prev=" << prev << " next=" << next << " last=" << last << "\n";

  while (true) {
    oracle::Tensor in;
    st = orch.transport().recv_tensor(prev, &in, 600000);
    if (!st) {
      if (st.code == oracle::Errc::disconnected) {
        std::cerr << "peer disconnected\n";
        return 1;
      }
      continue;
    }
    if (in.header.flags & oracle::kFlagLayerBlob) {
      orch.runner()->load_layer_blob(in.header.layer_id, in.payload);
      continue;
    }
    oracle::Tensor hidden;
    st = orch.runner()->forward(in.payload, kv, &hidden);
    if (!st) {
      std::cerr << "forward: " << st.message << "\n";
      continue;
    }
    hidden.header.seq_id = in.header.seq_id;
    hidden.header.token_id = in.header.token_id;
    if (last) {
      oracle::Tensor logits;
      st = orch.runner()->lm_head(hidden.payload, &logits);
      if (!st) {
        continue;
      }
      logits.header.seq_id = in.header.seq_id;
      const oracle::NodeId dest = master ? master->id : 0;
      orch.transport().send_tensor(dest, logits);
    } else {
      orch.transport().send_tensor(next, hidden);
    }
  }
}

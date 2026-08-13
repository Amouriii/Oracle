#include "oracle/orch/pipeline_orchestrator.hpp"
#include "oracle/runner/node_runner.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  oracle::ClusterConfig cfg;
  cfg.name = "tiny";
  cfg.model.name = "tiny-id";
  cfg.model.n_layers = 4;
  cfg.model.hidden_dim = 8;
  cfg.model.n_heads = 2;
  cfg.model.n_kv_heads = 2;
  cfg.model.head_dim = 4;
  cfg.model.n_vocab = 128;
  cfg.model.max_seq = 32;
  cfg.model.act_dtype = oracle::DType::F32;
  oracle::NodeConfig n;
  n.id = 0;
  n.role = "master";
  n.host = "127.0.0.1";
  n.ram_budget_gb = 8;
  cfg.nodes.push_back(n);

  oracle::PipelineOrchestrator orch;
  auto st = orch.init(cfg, oracle::make_accelerate_runner(), 0);
  if (!st) {
    std::cerr << "init " << st.message << "\n";
    return 1;
  }
  assert(orch.dag().size() == 1);
  assert(orch.dag()[0].is_embed);
  assert(orch.dag()[0].is_lm_head);

  oracle::AccelerateRunner metal_check;
  st = metal_check.load_layers(cfg.model, {0, 4}, {});
  assert(st.ok());
  std::vector<float> a(16), b(16), c(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      a[i * 4 + j] = (i == j) ? 1.f : 0.f;
      b[i * 4 + j] = static_cast<float>(i + j);
    }
  }
  st = metal_check.gemm_f32(4, 4, 4, a.data(), b.data(), c.data());
  assert(st.ok());
  assert(c[0] == b[0]);

  oracle::MetalNodeRunner metal;
  st = metal.load_layers(cfg.model, {0, 4}, {});
  if (st) {
    std::vector<float> mc(16);
    auto ms = metal.gemm_f32(4, 4, 4, a.data(), b.data(), mc.data());
    if (ms) {
      assert(std::abs(mc[5] - b[5]) < 1e-4f);
      std::cout << "metal gemm ok\n";
    } else {
      std::cout << "metal gemm skipped: " << ms.message << "\n";
    }
  }

  std::string acc;
  oracle::GenerateRequest req;
  req.prompt = "Hi";
  req.max_tokens = 4;
  req.stream = false;
  st = orch.generate(req, [&](const oracle::GenerateToken& t) { acc += t.text; });
  if (!st) {
    std::cerr << "generate " << st.message << "\n";
    return 1;
  }
  std::cout << "test_pipeline_tiny ok text_len=" << acc.size() << "\n";
  return 0;
}

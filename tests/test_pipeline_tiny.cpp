#include "oracle/orch/pipeline_orchestrator.hpp"
#include "oracle/runner/node_runner.hpp"

#include "check.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

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
  CHECK(orch.dag().size() == 1);
  CHECK(orch.dag()[0].is_embed);
  CHECK(orch.dag()[0].is_lm_head);

  oracle::AccelerateRunner metal_check;
  st = metal_check.load_layers(cfg.model, {0, 4}, {});
  CHECK(st.ok());
  std::vector<float> a(16), b(16), c(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      a[i * 4 + j] = (i == j) ? 1.f : 0.f;
      b[i * 4 + j] = static_cast<float>(i + j);
    }
  }
  st = metal_check.gemm_f32(4, 4, 4, a.data(), b.data(), c.data());
  CHECK(st.ok());
  CHECK(c[0] == b[0]);

  oracle::MetalNodeRunner metal;
  st = metal.load_layers(cfg.model, {0, 4}, {});
  if (st) {
    std::vector<float> mc(16);
    auto ms = metal.gemm_f32(4, 4, 4, a.data(), b.data(), mc.data());
    if (ms) {
      CHECK(std::abs(mc[5] - b[5]) < 1e-4f);
      std::cout << "metal gemm ok\n";
    } else {
      std::cout << "metal gemm skipped: " << ms.message << "\n";
    }
  }

  // A node started with no model configured falls back to this runner with the
  // cluster config's placeholder geometry -- 80 layers of 8192 hidden units.
  // Storing those identities densely would be ~43 GB, which is exactly how the
  // container image used to die of bad_alloc before serving a single request.
  {
    oracle::ModelMeta big;  // defaults are 70B-shaped on purpose
    CHECK(big.n_layers == 80);
    CHECK(big.hidden_dim == 8192);
    oracle::AccelerateRunner lean;
    CHECK_OK(lean.load_layers(big, {0, big.n_layers}, {}));

    oracle::KvLayout kvl;
    kvl.max_seq = 8;
    oracle::KvCache kvc;
    CHECK_OK(kvc.allocate(kvl));

    oracle::Tensor e, h, l;
    CHECK_OK(lean.embed(std::vector<int32_t>{3}, &e));
    CHECK_OK(lean.forward(e.payload, kvc, &h));
    CHECK_OK(lean.lm_head(h.payload, &l));
    CHECK(l.payload.size() == static_cast<size_t>(big.n_vocab) * 4);
    std::vector<float> logits(big.n_vocab);
    std::memcpy(logits.data(), l.payload.data(), logits.size() * 4);
    // Token 3 embeds to a one-hot at index 3 and passes through unchanged.
    CHECK(std::abs(logits[3] - 1.0f) < 1e-3f);
    CHECK(std::abs(logits[4]) < 1e-3f);
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

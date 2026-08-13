#include "oracle/orch/pipeline_orchestrator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <thread>

namespace oracle {

PipelineOrchestrator::~PipelineOrchestrator() { stop(); }

Status PipelineOrchestrator::init(ClusterConfig cfg, std::unique_ptr<NodeRunner> runner, NodeId self) {
  cfg_ = std::move(cfg);
  self_ = self;
  runner_ = std::move(runner);
  shards_ = std::make_unique<MemoryShardManager>(cfg_);
  auto st = shards_->assign_contiguous_layers();
  if (!st) {
    return st;
  }
  cfg_ = shards_->config();
  st = build_dag();
  if (!st) {
    return st;
  }
  const auto* me = cfg_.find(self_);
  if (!me) {
    return Status::fail(Errc::not_found, "self not in cluster");
  }
  if (!runner_) {
    runner_ = make_accelerate_runner();
  }
  st = runner_->load_layers(cfg_.model, me->layers, cfg_.model.path);
  if (!st) {
    return st;
  }
  auto kv = plan_kv(cfg_.model, me->layers);
  st = kv_.allocate(kv);
  if (!st) {
    return st;
  }
  for (const auto& n : cfg_.nodes) {
    HeartbeatState hs;
    hs.id = n.id;
    hs.last_seen = std::chrono::steady_clock::now();
    hs.alive = true;
    hb_[n.id] = hs;
  }
  running_.store(true);
  return Status::OK();
}

Status PipelineOrchestrator::build_dag() {
  dag_ = build_linear_dag(cfg_);
  if (dag_.empty()) {
    return Status::fail(Errc::invalid_argument, "empty dag");
  }
  return Status::OK();
}

Status PipelineOrchestrator::start_transport() {
  const auto* me = cfg_.find(self_);
  if (!me) {
    return Status::fail(Errc::not_found, "self");
  }
  TransportOptions opt;
  opt.bind_host = "0.0.0.0";
  opt.port = me->transport_port;
  opt.heartbeat_port = me->heartbeat_port;
  opt.mtu = cfg_.mtu;
  opt.local_payload_bytes = std::max<uint64_t>(cfg_.model.hidden_dim * 4ull, 8192ull * 2ull);
  auto st = tx_.listen(opt);
  if (!st) {
    return st;
  }
  return Status::OK();
}

Status PipelineOrchestrator::start_heartbeat() {
  const auto* me = cfg_.find(self_);
  std::vector<std::pair<std::string, uint16_t>> peers;
  for (const auto& n : cfg_.nodes) {
    if (n.id != self_) {
      peers.emplace_back(n.host, n.heartbeat_port);
    }
  }
  TransportOptions opt = {};
  opt.heartbeat_port = me ? me->heartbeat_port : cfg_.heartbeat_port;
  // heartbeat socket is created inside start_heartbeat using opt_ on transport; set via listen or patch:
  auto st = tx_.start_heartbeat(self_, peers);
  tx_.set_recv_handler([this](NodeId from, Tensor&& t) {
    if (t.header.flags & kFlagHeartbeat) {
      on_heartbeat(from);
    }
  });
  return st;
}

void PipelineOrchestrator::stop() {
  running_.store(false);
  tx_.stop();
}

Status PipelineOrchestrator::local_forward(const Tensor& in, Tensor* out) {
  const auto* me = cfg_.find(self_);
  bool last = me && dag_.empty() ? true : (!dag_.empty() && dag_.back().id == self_);
  Tensor hidden = in;
  auto st = runner_->forward(in.payload, kv_, &hidden);
  if (!st) {
    return st;
  }
  if (last) {
    return runner_->lm_head(hidden.payload, out);
  }
  *out = std::move(hidden);
  return Status::OK();
}

Status PipelineOrchestrator::tensor_ring_once(Tensor t, Tensor* out, int timeout_ms) {
  if (cfg_.nodes.size() <= 1) {
    return local_forward(t, out);
  }
  NodeId next = cfg_.nodes.front().id;
  for (size_t i = 0; i < cfg_.nodes.size(); ++i) {
    if (cfg_.nodes[i].id == self_ && i + 1 < cfg_.nodes.size()) {
      next = cfg_.nodes[i + 1].id;
      break;
    }
  }
  auto st = tx_.send_tensor(next, t);
  if (!st) {
    return st;
  }
  const NodeId last = cfg_.nodes.back().id;
  return tx_.recv_tensor(last, out, timeout_ms);
}

uint32_t PipelineOrchestrator::greedy_sample(std::span<const float> logits) const {
  if (logits.empty()) {
    return 0;
  }
  size_t best = 0;
  float v = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > v) {
      v = logits[i];
      best = i;
    }
  }
  return static_cast<uint32_t>(best);
}

std::vector<int32_t> PipelineOrchestrator::tokenize(std::string_view text) const {
  std::vector<int32_t> toks;
  toks.reserve(text.size());
  for (unsigned char c : text) {
    toks.push_back(static_cast<int32_t>(c) % std::max<uint32_t>(2, cfg_.model.n_vocab));
  }
  if (toks.empty()) {
    toks.push_back(1);
  }
  return toks;
}

std::string PipelineOrchestrator::detokenize(uint32_t token) const {
  if (token < 128) {
    char c = static_cast<char>(token);
    if (c >= 32) {
      return std::string(1, c);
    }
  }
  return " ";
}

Status PipelineOrchestrator::generate(const GenerateRequest& req,
                                      const std::function<void(const GenerateToken&)>& on_token) {
  tick_heartbeats();
  for (const auto& n : cfg_.nodes) {
    if (n.id != self_ && !worker_alive(n.id) && cfg_.nodes.size() > 1) {
      return Status::fail(Errc::worker_dead, "worker " + std::to_string(n.id) + " missed heartbeats");
    }
  }
  kv_.reset();
  auto toks = tokenize(req.prompt);
  Tensor hidden;
  auto st = runner_->embed(toks, &hidden);
  if (!st) {
    return st;
  }
  hidden.header.seq_id = req.seq_id ? req.seq_id : next_seq_.fetch_add(1);
  hidden.header.flags = kFlagPrefill;
  hidden.header.checksum = crc32(hidden.payload);

  Tensor cur = hidden;
  const bool single = cfg_.nodes.size() <= 1 || (cfg_.find(self_) && cfg_.find(self_)->role == "master" &&
                                                 dag_.size() == 1);
  for (uint32_t step = 0; step < req.max_tokens; ++step) {
    Tensor next;
    if (single) {
      st = local_forward(cur, &next);
    } else {
      st = tensor_ring_once(cur, &next, 5000);
      if (!st) {
        st = local_forward(cur, &next);
      }
    }
    if (!st) {
      return st;
    }
    uint32_t tok = 0;
    if (next.header.is_logits() || next.header.dtype_enum() == DType::F32) {
      const size_t n = next.payload.size() / 4;
      std::vector<float> logits(n);
      if (n) {
        std::memcpy(logits.data(), next.payload.data(), n * 4);
      }
      tok = greedy_sample(logits);
    }
    GenerateToken gt;
    gt.token_id = tok;
    gt.text = detokenize(tok);
    gt.stop = (step + 1 == req.max_tokens) || tok == 0;
    if (on_token) {
      on_token(gt);
    }
    if (gt.stop) {
      break;
    }
    std::vector<int32_t> one{static_cast<int32_t>(tok)};
    st = runner_->embed(one, &cur);
    if (!st) {
      return st;
    }
    cur.header.flags = kFlagDecode;
    cur.header.token_id = tok;
    cur.header.seq_id = hidden.header.seq_id;
  }
  return Status::OK();
}

Status PipelineOrchestrator::serve_openai(uint16_t port) { return run_openai_server(*this, port); }

}  // namespace oracle

#include "oracle/orch/pipeline_orchestrator.hpp"

#include "oracle/compute/blas.hpp"
#include "oracle/model/quant.hpp"
#include "oracle/runner/gguf_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

namespace oracle {
namespace {

constexpr const char* kReleaseOp = "RELEASE";

std::string json_escape(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

uint64_t splitmix64(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

}  // namespace

std::vector<DagNode> build_linear_dag(const ClusterConfig& cfg) {
  std::vector<DagNode> dag;
  dag.reserve(cfg.nodes.size());
  for (size_t i = 0; i < cfg.nodes.size(); ++i) {
    DagNode n;
    n.id = cfg.nodes[i].id;
    n.layers = cfg.nodes[i].layers;
    n.is_embed = (i == 0);
    n.is_lm_head = (i + 1 == cfg.nodes.size());
    dag.push_back(n);
  }
  return dag;
}

PipelineOrchestrator::~PipelineOrchestrator() { stop(); }

bool PipelineOrchestrator::is_master() const {
  const auto* me = cfg_.find(self_);
  return me && me->role == "master";
}

const model::Tokenizer* PipelineOrchestrator::tokenizer() const {
  if (const auto* g = dynamic_cast<const GgufRunner*>(runner_.get())) {
    return &g->tokenizer();
  }
  if (const auto* l = dynamic_cast<const LlamaCppRunner*>(runner_.get())) {
    if (const auto* g = dynamic_cast<const GgufRunner*>(l->backend())) {
      return &g->tokenizer();
    }
  }
  return nullptr;
}

uint64_t PipelineOrchestrator::uptime_seconds() const {
  if (started_at_.time_since_epoch().count() == 0) {
    return 0;
  }
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - started_at_)
          .count());
}

Status PipelineOrchestrator::init(ClusterConfig cfg, std::unique_ptr<NodeRunner> runner, NodeId self) {
  cfg_ = std::move(cfg);
  self_ = self;
  runner_ = std::move(runner);
  started_at_ = std::chrono::steady_clock::now();

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
    return Status::fail(Errc::not_found,
                        "node " + std::to_string(self_) + " is not listed in the cluster config");
  }
  if (!runner_) {
    runner_ = make_accelerate_runner();
  }

  if (!cfg_.model.path.empty()) {
    // Recognise the model before loading so sharding, the KV plan and /v1/models
    // all reflect the file rather than whatever the config guessed.
    model::ModelInfo info;
    auto probe = model::inspect_gguf(cfg_.model.path, &info);
    if (probe) {
      const uint32_t configured_max_seq = cfg_.model.max_seq;
      model::apply_to_model_meta(info, &cfg_.model);
      if (configured_max_seq && configured_max_seq < cfg_.model.max_seq) {
        cfg_.model.max_seq = configured_max_seq;
      }
      shards_->config().model = cfg_.model;
      st = shards_->assign_contiguous_layers();
      if (!st) {
        return st;
      }
      cfg_ = shards_->config();
      st = build_dag();
      if (!st) {
        return st;
      }
      me = cfg_.find(self_);
      model_name_ = info.name.empty() ? cfg_.model.name : info.name;
    } else {
      return Status::fail(probe.code, "model: " + probe.message);
    }
  } else {
    model_name_ = cfg_.model.name.empty() ? "oracle" : cfg_.model.name;
  }

  st = runner_->load_layers(cfg_.model, me->layers, cfg_.model.path);
  if (!st) {
    return st;
  }

  kv_layout_ = plan_kv(cfg_.model, me->layers);
  max_sequences_ = std::max<size_t>(1, cfg_.max_sequences);

  RegistryConfig rc;
  rc.heartbeat_interval_ms = cfg_.heartbeat_interval_ms;
  rc.heartbeat_misses = cfg_.heartbeat_misses;
  registry_.configure(rc);
  registry_.seed(cfg_, self_);

  std::vector<LayerRange> stages;
  for (const auto& n : cfg_.nodes) {
    stages.push_back(n.layers);
  }
  scheduler_.attach_registry(&registry_);
  scheduler_.set_required_stages(std::move(stages));
  scheduler_.set_requirement(requirement());

  for (const auto& n : cfg_.nodes) {
    HeartbeatState hs;
    hs.id = n.id;
    hs.last_seen = std::chrono::steady_clock::now();
    hs.alive = n.id == self_;
    hb_[n.id] = hs;
  }
  publish_local_status();
  running_.store(true);
  return Status::OK();
}

Status PipelineOrchestrator::configure_security(const security::SecurityConfig& cfg) {
  auto st = security_.configure(cfg);
  if (!st) {
    return st;
  }
  if (cfg.verify_model_integrity && !cfg_.model.path.empty()) {
    st = security_.verify_model_file(cfg_.model.path);
    if (!st) {
      return st;
    }
  }
  return Status::OK();
}

Status PipelineOrchestrator::configure_scheduler(const SchedulerConfig& cfg) {
  scheduler_.configure(cfg);
  return Status::OK();
}

Status PipelineOrchestrator::build_dag() {
  dag_ = build_linear_dag(cfg_);
  if (dag_.empty()) {
    return Status::fail(Errc::invalid_argument, "cluster config lists no nodes");
  }
  return Status::OK();
}

ResourceRequirement PipelineOrchestrator::requirement() const {
  ResourceRequirement r;
  r.model = cfg_.model.name;
  r.ram_bytes = cfg_.model.bytes_per_layer;
  r.kv_bytes = kv_layout_.bytes_total;
  return r;
}

NodeId PipelineOrchestrator::next_hop() const {
  for (size_t i = 0; i < cfg_.nodes.size(); ++i) {
    if (cfg_.nodes[i].id == self_ && i + 1 < cfg_.nodes.size()) {
      return cfg_.nodes[i + 1].id;
    }
  }
  return self_;
}

NodeId PipelineOrchestrator::last_stage() const {
  return cfg_.nodes.empty() ? self_ : cfg_.nodes.back().id;
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

  // Only nodes that can present a valid signature over a fresh nonce join.
  tx_.set_auth_handler([this](const Handshake& h, std::string* reason) {
    if (!security_.config().require_worker_auth) {
      registry_.note_join(h.node_id, {}, h.runner, {h.layer_start, h.layer_end});
      return true;
    }
    if (!security_.verify(h.node_id, h.nonce, h.signature)) {
      if (reason) {
        *reason = "signature does not match the cluster secret";
      }
      security_.audit().alert("worker", "node-" + std::to_string(h.node_id),
                              "rejected: bad handshake signature");
      return false;
    }
    if (!cfg_.find(h.node_id)) {
      if (reason) {
        *reason = "node id is not in the cluster config";
      }
      security_.audit().alert("worker", "node-" + std::to_string(h.node_id),
                              "rejected: unknown node id");
      return false;
    }
    registry_.note_join(h.node_id, {}, h.runner, {h.layer_start, h.layer_end});
    security_.audit().info("worker", "node-" + std::to_string(h.node_id),
                           "joined with layers [" + std::to_string(h.layer_start) + ", " +
                               std::to_string(h.layer_end) + ")");
    return true;
  });
  return Status::OK();
}

Status PipelineOrchestrator::start_heartbeat() {
  std::vector<std::pair<std::string, uint16_t>> peers;
  for (const auto& n : cfg_.nodes) {
    if (n.id != self_) {
      peers.emplace_back(n.host, n.heartbeat_port);
    }
  }
  tx_.set_recv_handler([this](NodeId from, Tensor&& t) {
    if (t.header.flags & kFlagHeartbeat) {
      on_heartbeat(from, std::string_view(reinterpret_cast<const char*>(t.payload.data()),
                                          t.payload.size()));
    }
  });
  auto st = tx_.start_heartbeat(self_, peers);
  if (!st) {
    return st;
  }
  publish_local_status();
  if (!reliability_.joinable()) {
    reliability_ = std::thread([this] { reliability_loop(); });
  }
  return Status::OK();
}

void PipelineOrchestrator::publish_local_status() {
  const auto* me = cfg_.find(self_);
  auto w = sample_local_resources(self_, me ? me->host : "127.0.0.1", me ? me->role : "worker");
  w.layers = me ? me->layers : LayerRange{};
  w.runner = runner_ ? runner_->name() : "none";
  w.model = cfg_.model.name;
  w.max_concurrent = std::max<uint32_t>(1, scheduler_.config().max_concurrent);
  {
    std::lock_guard<std::mutex> g(kv_mu_);
    w.active_requests = static_cast<uint32_t>(kv_.size());
  }
  if (const auto* g = dynamic_cast<const GgufRunner*>(runner_.get())) {
    w.resident_weight_bytes = g->resident_weight_bytes();
  }
  const auto stats = scheduler_.stats();
  w.queue_depth = stats.queued;
  if (is_master()) {
    w.active_requests = stats.running;
  }
  registry_.upsert(w);
  tx_.set_heartbeat_payload(w.encode_heartbeat());
}

void PipelineOrchestrator::stop() {
  running_.store(false);
  scheduler_.shutdown();
  tx_.stop();
  if (router_.joinable()) {
    router_.join();
  }
  if (reliability_.joinable()) {
    reliability_.join();
  }
  std::lock_guard<std::mutex> g(chan_mu_);
  for (auto& [seq, ch] : channels_) {
    std::lock_guard<std::mutex> cg(ch->mu);
    ch->failed = true;
    ch->error = "orchestrator stopped";
    ch->ready = true;
    ch->cv.notify_all();
  }
}

Status PipelineOrchestrator::connect_peers() {
  if (single_node()) {
    return Status::OK();
  }
  const auto* me = cfg_.find(self_);
  if (!me) {
    return Status::fail(Errc::not_found, "self");
  }

  Handshake hello;
  hello.node_id = self_;
  hello.role = me->role;
  hello.runner = runner_ ? runner_->name() : "none";
  hello.layer_start = me->layers.start;
  hello.layer_end = me->layers.end;

  if (is_master()) {
    // The master dials every other node so a worker never needs an inbound port
    // reachable from anywhere but its previous hop.
    for (const auto& n : cfg_.nodes) {
      if (n.id == self_) {
        continue;
      }
      auto st = tx_.ensure_connected(n.id, n.host, n.transport_port, 15000);
      if (!st) {
        registry_.mark_dead(n.id, st.message);
        security_.audit().warn("worker", "node-" + std::to_string(n.id), st.message);
        return Status::fail(st.code, "connecting to node " + std::to_string(n.id) + ": " + st.message);
      }
      hello.nonce = security_.new_nonce();
      hello.signature = security_.sign(self_, hello.nonce);
      st = tx_.register_with(n.id, hello);
      if (!st) {
        registry_.mark_dead(n.id, st.message);
        return Status::fail(st.code, "registering with node " + std::to_string(n.id) + ": " + st.message);
      }
      registry_.note_join(n.id, n.host, {}, n.layers);
    }
    if (!router_.joinable()) {
      router_ = std::thread([this] { router_loop(); });
    }
    return Status::OK();
  }

  // A worker waits for the master's registration, then dials the node it must
  // send its activations to.
  Handshake peer;
  auto st = tx_.accept_registration(60000, &peer);
  if (!st) {
    return Status::fail(st.code, "waiting for the master to register: " + st.message);
  }
  registry_.note_join(peer.node_id, {}, peer.runner, {peer.layer_start, peer.layer_end});

  const NodeId next = next_hop();
  if (next != self_) {
    const auto* n = cfg_.find(next);
    if (!n) {
      return Status::fail(Errc::not_found, "next hop " + std::to_string(next) + " is not configured");
    }
    st = tx_.ensure_connected(next, n->host, n->transport_port, 30000);
    if (!st) {
      return Status::fail(st.code, "connecting to the next hop: " + st.message);
    }
    hello.nonce = security_.new_nonce();
    hello.signature = security_.sign(self_, hello.nonce);
    st = tx_.register_with(next, hello);
    if (!st) {
      return st;
    }
  } else {
    // Last stage: logits go straight back to the master.
    const auto* master = cfg_.master();
    if (master && master->id != self_ && !tx_.connected(master->id)) {
      st = tx_.ensure_connected(master->id, master->host, master->transport_port, 30000);
      if (!st) {
        return Status::fail(st.code, "connecting back to the master: " + st.message);
      }
    }
  }
  return Status::OK();
}

void PipelineOrchestrator::router_loop() {
  // Master side: returning logits are matched to the request that is waiting on
  // that sequence id, so concurrent requests never steal each other's frames.
  while (running_.load()) {
    NodeId from = 0;
    Tensor t;
    auto st = tx_.recv_any(&from, &t, 200);
    if (!st) {
      if (st.code == Errc::disconnected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      continue;
    }
    if (t.header.flags & kFlagControl) {
      continue;
    }
    std::shared_ptr<SeqChannel> ch;
    {
      std::lock_guard<std::mutex> g(chan_mu_);
      const auto it = channels_.find(t.header.seq_id);
      if (it != channels_.end()) {
        ch = it->second;
      }
    }
    if (!ch) {
      continue;  // a cancelled or timed-out request; drop the late frame
    }
    std::lock_guard<std::mutex> cg(ch->mu);
    ch->payload = std::move(t);
    ch->ready = true;
    ch->cv.notify_all();
  }
}

void PipelineOrchestrator::reliability_loop() {
  // Ages out silent workers, retries dead links and republishes local load.
  while (running_.load()) {
    for (int i = 0; i < 10 && running_.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!running_.load()) {
      break;
    }
    tick_heartbeats();
    publish_local_status();

    const auto dead = registry_.tick();
    for (NodeId id : dead) {
      security_.audit().alert("worker", "node-" + std::to_string(id),
                              "marked dead after missing heartbeats");
      tx_.disconnect(id);
    }
    if (!is_master()) {
      continue;
    }
    for (NodeId id : registry_.due_for_reconnect()) {
      const auto* n = cfg_.find(id);
      if (!n || id == self_) {
        continue;
      }
      auto st = tx_.ensure_connected(id, n->host, n->transport_port, 500);
      if (!st) {
        continue;
      }
      Handshake hello;
      hello.node_id = self_;
      hello.role = "master";
      hello.runner = runner_ ? runner_->name() : "none";
      const auto* me = cfg_.find(self_);
      hello.layer_start = me ? me->layers.start : 0;
      hello.layer_end = me ? me->layers.end : 0;
      hello.nonce = security_.new_nonce();
      hello.signature = security_.sign(self_, hello.nonce);
      if (tx_.register_with(id, hello)) {
        registry_.note_reconnect(id);
        security_.audit().info("worker", "node-" + std::to_string(id), "reconnected");
      }
    }
  }
}

KvCache* PipelineOrchestrator::kv_for(uint64_t seq_id, bool reset) {
  std::lock_guard<std::mutex> g(kv_mu_);
  auto it = kv_.find(seq_id);
  if (it == kv_.end()) {
    if (kv_.size() >= max_sequences_ && !kv_lru_.empty()) {
      // Oldest sequence loses its cache; a request that comes back later simply
      // re-prefills rather than growing memory without bound.
      const uint64_t victim = kv_lru_.front();
      kv_lru_.erase(kv_lru_.begin());
      kv_.erase(victim);
    }
    auto cache = std::make_unique<KvCache>();
    if (!cache->allocate(kv_layout_)) {
      return nullptr;
    }
    it = kv_.emplace(seq_id, std::move(cache)).first;
    kv_lru_.push_back(seq_id);
  } else {
    const auto pos = std::find(kv_lru_.begin(), kv_lru_.end(), seq_id);
    if (pos != kv_lru_.end()) {
      kv_lru_.erase(pos);
    }
    kv_lru_.push_back(seq_id);
  }
  if (reset) {
    it->second->reset();
  }
  return it->second.get();
}

void PipelineOrchestrator::drop_sequence(uint64_t seq_id) {
  std::lock_guard<std::mutex> g(kv_mu_);
  kv_.erase(seq_id);
  const auto pos = std::find(kv_lru_.begin(), kv_lru_.end(), seq_id);
  if (pos != kv_lru_.end()) {
    kv_lru_.erase(pos);
  }
}

void PipelineOrchestrator::release_sequence(uint64_t seq_id) {
  drop_sequence(seq_id);
  if (single_node()) {
    return;
  }
  const NodeId next = next_hop();
  if (next == self_) {
    return;
  }
  TensorHeader hdr{};
  hdr.flags = kFlagControl | kFlagEos;
  hdr.seq_id = seq_id;
  hdr.dtype = static_cast<uint16_t>(DType::I8);
  hdr.rank = 1;
  const std::string op = kReleaseOp;
  hdr.shape[0] = static_cast<uint32_t>(op.size());
  hdr.nbytes = op.size();
  auto payload = std::span<const std::byte>(reinterpret_cast<const std::byte*>(op.data()), op.size());
  hdr.checksum = crc32(payload);
  (void)tx_.send_tensor(next, hdr, payload);
}

Status PipelineOrchestrator::local_forward(const Tensor& in, KvCache& kv, Tensor* out,
                                           bool want_logits) {
  Tensor hidden;
  auto st = runner_->forward(in.payload, kv, &hidden);
  if (!st) {
    return st;
  }
  hidden.header.seq_id = in.header.seq_id;
  hidden.header.token_id = in.header.token_id;
  hidden.header.flags |= (in.header.flags & (kFlagPrefill | kFlagDecode));
  if (!want_logits) {
    *out = std::move(hidden);
    return Status::OK();
  }
  st = runner_->lm_head(hidden.payload, out);
  if (!st) {
    return st;
  }
  out->header.seq_id = in.header.seq_id;
  return Status::OK();
}

Status PipelineOrchestrator::run_stage(Tensor in, uint64_t seq_id, KvCache& kv, Tensor* logits,
                                       int timeout_ms) {
  in.header.seq_id = seq_id;
  if (single_node()) {
    return local_forward(in, kv, logits, true);
  }

  // This node's own layers first, then hand the activations to the next hop.
  Tensor hidden;
  auto st = local_forward(in, kv, &hidden, false);
  if (!st) {
    return st;
  }
  hidden.header.seq_id = seq_id;

  auto ch = std::make_shared<SeqChannel>();
  {
    std::lock_guard<std::mutex> g(chan_mu_);
    channels_[seq_id] = ch;
  }
  struct Unregister {
    PipelineOrchestrator* self;
    uint64_t seq;
    ~Unregister() {
      std::lock_guard<std::mutex> g(self->chan_mu_);
      self->channels_.erase(seq);
    }
  } unreg{this, seq_id};

  const NodeId next = next_hop();
  st = tx_.send_tensor(next, hidden);
  if (!st) {
    registry_.mark_dead(next, st.message);
    return Status::fail(st.code, "sending activations to node " + std::to_string(next) + ": " + st.message);
  }

  std::unique_lock<std::mutex> lk(ch->mu);
  if (!ch->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return ch->ready; })) {
    return Status::fail(Errc::timeout, "no logits came back for sequence " + std::to_string(seq_id) +
                                           " within " + std::to_string(timeout_ms) + " ms");
  }
  if (ch->failed) {
    return Status::fail(Errc::io, ch->error);
  }
  *logits = std::move(ch->payload);
  return Status::OK();
}

std::vector<int32_t> PipelineOrchestrator::tokenize(const std::string& text, bool add_bos) const {
  if (const auto* tok = tokenizer(); tok && tok->n_vocab() > 0) {
    return tok->encode(text, add_bos && tok->add_bos_default());
  }
  // No vocabulary (identity runner / no model file): byte tokens keep the
  // transport and orchestration paths exercisable.
  std::vector<int32_t> out;
  out.reserve(text.size() + 1);
  const uint32_t vocab = std::max<uint32_t>(2, cfg_.model.n_vocab);
  for (unsigned char c : text) {
    out.push_back(static_cast<int32_t>(c % vocab));
  }
  if (out.empty()) {
    out.push_back(1);
  }
  return out;
}

std::string PipelineOrchestrator::detokenize(int32_t token) const {
  if (const auto* tok = tokenizer(); tok && tok->n_vocab() > 0) {
    return tok->decode(token);
  }
  if (token >= 32 && token < 127) {
    return std::string(1, static_cast<char>(token));
  }
  return {};
}

std::string PipelineOrchestrator::render_prompt(const GenerateRequest& req) const {
  if (req.messages.empty()) {
    return req.prompt;
  }
  if (const auto* tok = tokenizer(); tok && tok->n_vocab() > 0) {
    return tok->apply_chat_template(req.messages, true);
  }
  std::string out;
  for (const auto& m : req.messages) {
    out += m.role + ": " + m.content + "\n";
  }
  out += "assistant:";
  return out;
}

uint32_t PipelineOrchestrator::sample(std::span<const float> logits, const GenerateRequest& req,
                                      const std::vector<int32_t>& history, uint64_t* rng_state) const {
  if (logits.empty()) {
    return 0;
  }
  std::vector<float> work(logits.begin(), logits.end());

  if (req.repeat_penalty > 1.0f && !history.empty()) {
    const size_t window = std::min<size_t>(history.size(), 64);
    for (size_t i = history.size() - window; i < history.size(); ++i) {
      const int32_t t = history[i];
      if (t < 0 || static_cast<size_t>(t) >= work.size()) {
        continue;
      }
      work[static_cast<size_t>(t)] /= (work[static_cast<size_t>(t)] > 0 ? req.repeat_penalty
                                                                        : 1.0f / req.repeat_penalty);
    }
  }

  if (req.temperature <= 0.0f) {
    return static_cast<uint32_t>(std::max_element(work.begin(), work.end()) - work.begin());
  }

  std::vector<uint32_t> idx(work.size());
  std::iota(idx.begin(), idx.end(), 0u);
  const size_t k = req.top_k ? std::min<size_t>(req.top_k, idx.size()) : idx.size();
  std::partial_sort(idx.begin(), idx.begin() + static_cast<long>(k), idx.end(),
                    [&](uint32_t a, uint32_t b) { return work[a] > work[b]; });
  idx.resize(k);

  const float mx = work[idx.front()];
  std::vector<float> probs(k);
  float sum = 0.f;
  for (size_t i = 0; i < k; ++i) {
    probs[i] = std::exp((work[idx[i]] - mx) / req.temperature);
    sum += probs[i];
  }
  if (sum <= 0.f) {
    return idx.front();
  }
  for (auto& p : probs) {
    p /= sum;
  }

  size_t cutoff = k;
  if (req.top_p > 0.f && req.top_p < 1.0f) {
    float acc = 0.f;
    for (size_t i = 0; i < k; ++i) {
      acc += probs[i];
      if (acc >= req.top_p) {
        cutoff = i + 1;
        break;
      }
    }
  }
  float renorm = 0.f;
  for (size_t i = 0; i < cutoff; ++i) {
    renorm += probs[i];
  }
  const double r = static_cast<double>(splitmix64(rng_state) >> 11) / static_cast<double>(1ull << 53);
  double acc = 0.0;
  for (size_t i = 0; i < cutoff; ++i) {
    acc += probs[i] / renorm;
    if (r <= acc) {
      return idx[i];
    }
  }
  return idx[cutoff ? cutoff - 1 : 0];
}

Status PipelineOrchestrator::generate(const GenerateRequest& req,
                                      const std::function<void(const GenerateToken&)>& on_token,
                                      GenerateResult* result) {
  if (!runner_) {
    return Status::fail(Errc::invalid_argument, "no runner is loaded");
  }
  tick_heartbeats();
  for (const auto& n : cfg_.nodes) {
    if (n.id != self_ && !registry_.healthy(n.id)) {
      return Status::fail(Errc::worker_dead,
                          "node " + std::to_string(n.id) + " is not healthy; refusing to dispatch");
    }
  }

  const std::string prompt = render_prompt(req);
  const auto tokens = tokenize(prompt, true);
  if (tokens.empty()) {
    return Status::fail(Errc::invalid_argument, "prompt tokenised to nothing");
  }
  if (cfg_.model.max_seq && tokens.size() + req.max_tokens > cfg_.model.max_seq) {
    return Status::fail(Errc::invalid_argument,
                        "prompt (" + std::to_string(tokens.size()) + " tokens) plus max_tokens (" +
                            std::to_string(req.max_tokens) + ") exceeds the " +
                            std::to_string(cfg_.model.max_seq) + " token context window");
  }

  const uint64_t seq_id = req.seq_id ? req.seq_id : scheduler_.next_seq_id();
  KvCache* kv = kv_for(seq_id, true);
  if (!kv) {
    return Status::fail(Errc::oom, "could not allocate a KV cache for this request");
  }
  struct Release {
    PipelineOrchestrator* self;
    uint64_t seq;
    ~Release() { self->release_sequence(seq); }
  } release{this, seq_id};

  uint64_t rng = req.seed ? req.seed : (0x9E3779B97F4A7C15ull ^ seq_id);
  const model::Tokenizer* tok = tokenizer();
  const int stage_timeout_ms =
      static_cast<int>(std::max<uint32_t>(5000, scheduler_.config().request_timeout_ms));

  GenerateResult res;
  res.prompt_tokens = static_cast<uint32_t>(tokens.size());
  std::string produced;
  std::vector<int32_t> history(tokens.begin(), tokens.end());

  // ---- prefill: the whole prompt travels the mesh as one batched frame -----
  const auto prefill_start = std::chrono::steady_clock::now();
  Tensor embedded;
  auto st = runner_->embed(tokens, &embedded);
  if (!st) {
    return st;
  }
  embedded.header.seq_id = seq_id;
  embedded.header.flags |= kFlagPrefill;
  embedded.header.checksum = crc32(embedded.payload);

  Tensor logits;
  st = run_stage(std::move(embedded), seq_id, *kv, &logits, stage_timeout_ms);
  if (!st) {
    return st;
  }
  res.prefill_ms = ms_since(prefill_start);

  const auto decode_start = std::chrono::steady_clock::now();
  for (uint32_t step = 0; step < req.max_tokens; ++step) {
    if (logits.payload.size() < 4) {
      return Status::fail(Errc::protocol, "the last stage returned no logits");
    }
    std::vector<float> lv(logits.payload.size() / 4);
    std::memcpy(lv.data(), logits.payload.data(), lv.size() * 4);

    const uint32_t token = sample(lv, req, history, &rng);
    history.push_back(static_cast<int32_t>(token));

    const bool eog = tok ? tok->is_eog(static_cast<int32_t>(token)) : token == 0;
    std::string piece = eog ? std::string{} : detokenize(static_cast<int32_t>(token));
    produced += piece;

    bool hit_stop = false;
    for (const auto& s : req.stop) {
      if (!s.empty() && produced.size() >= s.size() &&
          produced.compare(produced.size() - s.size(), s.size(), s) == 0) {
        produced.erase(produced.size() - s.size());
        hit_stop = true;
        break;
      }
    }

    GenerateToken gt;
    gt.token_id = token;
    gt.text = std::move(piece);
    gt.index = step;
    gt.stop = eog || hit_stop || (step + 1 == req.max_tokens);
    ++res.completion_tokens;
    if (on_token && !gt.text.empty()) {
      on_token(gt);
    } else if (on_token && gt.stop) {
      GenerateToken final_tok = gt;
      final_tok.text.clear();
      on_token(final_tok);
    }
    if (eog || hit_stop) {
      res.finish_reason = "stop";
      break;
    }
    if (step + 1 == req.max_tokens) {
      res.finish_reason = "length";
      break;
    }

    Tensor next;
    st = runner_->embed(std::vector<int32_t>{static_cast<int32_t>(token)}, &next);
    if (!st) {
      return st;
    }
    next.header.seq_id = seq_id;
    next.header.token_id = token;
    next.header.flags |= kFlagDecode;
    next.header.checksum = crc32(next.payload);
    st = run_stage(std::move(next), seq_id, *kv, &logits, stage_timeout_ms);
    if (!st) {
      return st;
    }
  }

  res.decode_ms = ms_since(decode_start);
  if (res.decode_ms > 0) {
    res.tokens_per_second = static_cast<double>(res.completion_tokens) / (res.decode_ms / 1000.0);
  }
  res.text = produced;
  requests_served_.fetch_add(1, std::memory_order_relaxed);
  tokens_generated_.fetch_add(res.completion_tokens, std::memory_order_relaxed);
  if (result) {
    *result = std::move(res);
  }
  return Status::OK();
}

Status PipelineOrchestrator::run_worker_loop() {
  const auto* me = cfg_.find(self_);
  if (!me) {
    return Status::fail(Errc::not_found, "self");
  }
  const NodeId next = next_hop();
  const bool last = next == self_;
  const auto* master = cfg_.master();
  const NodeId reply_to = master ? master->id : 0;

  while (running_.load()) {
    NodeId from = 0;
    Tensor in;
    auto st = tx_.recv_any(&from, &in, 250);
    if (!st) {
      if (st.code == Errc::timeout || st.code == Errc::not_found) {
        continue;
      }
      if (st.code == Errc::disconnected) {
        security_.audit().warn("worker", "node-" + std::to_string(from), "peer disconnected");
        tx_.disconnect(from);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      security_.audit().warn("worker", "node-" + std::to_string(from), st.message);
      continue;
    }

    if (in.header.flags & kFlagControl) {
      const std::string op(reinterpret_cast<const char*>(in.payload.data()), in.payload.size());
      if (op == kReleaseOp) {
        drop_sequence(in.header.seq_id);
        if (!last) {
          (void)tx_.send_tensor(next, in);
        }
      }
      continue;
    }
    if (in.header.flags & kFlagLayerBlob) {
      (void)runner_->load_layer_blob(in.header.layer_id, in.payload);
      continue;
    }

    KvCache* kv = kv_for(in.header.seq_id, (in.header.flags & kFlagPrefill) != 0);
    if (!kv) {
      security_.audit().alert("worker", "seq-" + std::to_string(in.header.seq_id),
                              "could not allocate a KV cache");
      continue;
    }

    Tensor out;
    st = local_forward(in, *kv, &out, last);
    if (!st) {
      security_.audit().warn("worker", "seq-" + std::to_string(in.header.seq_id),
                             "forward failed: " + st.message);
      continue;
    }
    out.header.seq_id = in.header.seq_id;
    out.header.token_id = in.header.token_id;
    const NodeId dest = last ? reply_to : next;
    st = tx_.send_tensor(dest, out);
    if (!st) {
      security_.audit().warn("worker", "node-" + std::to_string(dest), "send failed: " + st.message);
      registry_.mark_dead(dest, st.message);
    }
    publish_local_status();
  }
  return Status::OK();
}

Status PipelineOrchestrator::benchmark_links(uint32_t payload_bytes, uint32_t iterations) {
  if (single_node()) {
    return Status::fail(Errc::invalid_argument, "nothing to benchmark on a single node");
  }
  std::vector<std::byte> payload(payload_bytes ? payload_bytes : 65536);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::byte>(i & 0xFF);
  }
  for (const auto& n : cfg_.nodes) {
    if (n.id == self_ || !tx_.connected(n.id)) {
      continue;
    }
    TensorHeader hdr{};
    hdr.flags = kFlagControl;
    hdr.dtype = static_cast<uint16_t>(DType::I8);
    hdr.rank = 1;
    hdr.shape[0] = static_cast<uint32_t>(payload.size());
    hdr.nbytes = payload.size();
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t sent = 0;
    for (uint32_t i = 0; i < std::max(1u, iterations); ++i) {
      hdr.seq_id = i;
      if (!tx_.send_tensor(n.id, hdr, payload)) {
        break;
      }
      ++sent;
    }
    const double elapsed_ms = ms_since(t0);
    if (sent == 0 || elapsed_ms <= 0) {
      continue;
    }
    const double gbps =
        (static_cast<double>(sent) * payload.size() * 8.0) / (elapsed_ms / 1000.0) / 1e9;
    registry_.set_link(n.id, elapsed_ms / sent, gbps);
  }
  return Status::OK();
}

void PipelineOrchestrator::on_heartbeat(NodeId id) { on_heartbeat(id, {}); }

void PipelineOrchestrator::on_heartbeat(NodeId id, std::string_view payload) {
  {
    std::lock_guard<std::mutex> g(hb_mu_);
    auto& s = hb_[id];
    s.id = id;
    s.last_seen = std::chrono::steady_clock::now();
    s.misses = 0;
    s.alive = true;
  }
  registry_.note_heartbeat(id, payload);
}

bool PipelineOrchestrator::worker_alive(NodeId id) const { return registry_.healthy(id); }

void PipelineOrchestrator::tick_heartbeats() {
  const auto now = std::chrono::steady_clock::now();
  const auto timeout =
      std::chrono::milliseconds(cfg_.heartbeat_interval_ms * std::max(1u, cfg_.heartbeat_misses));
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

std::string PipelineOrchestrator::health_json() const {
  const auto workers = registry_.snapshot();
  size_t alive = 0;
  for (const auto& w : workers) {
    if (w.healthy()) {
      ++alive;
    }
  }
  const bool degraded = alive < workers.size();
  std::ostringstream os;
  os << "{\"status\":\"" << (alive == 0 ? "down" : (degraded ? "degraded" : "ok")) << "\""
     << ",\"node\":" << self_ << ",\"role\":\"" << (is_master() ? "master" : "worker") << "\""
     << ",\"model\":\"" << json_escape(model_name_) << "\""
     << ",\"workers\":{\"total\":" << workers.size() << ",\"alive\":" << alive << "}"
     << ",\"uptime_seconds\":" << uptime_seconds() << "}";
  return os.str();
}

std::string PipelineOrchestrator::models_json() const {
  std::ostringstream os;
  os << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(model_name_)
     << "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"oracle\"";
  os << ",\"oracle\":{\"n_layers\":" << cfg_.model.n_layers
     << ",\"hidden_dim\":" << cfg_.model.hidden_dim << ",\"n_heads\":" << cfg_.model.n_heads
     << ",\"n_kv_heads\":" << cfg_.model.n_kv_heads << ",\"n_vocab\":" << cfg_.model.n_vocab
     << ",\"context_length\":" << cfg_.model.max_seq
     << ",\"weight_bytes\":" << cfg_.model.total_weight_bytes;
  if (const auto* g = dynamic_cast<const GgufRunner*>(runner_.get())) {
    os << ",\"quantization\":\"" << json_escape(g->info().quantization) << "\""
       << ",\"architecture\":\"" << json_escape(g->info().architecture) << "\""
       << ",\"parameters\":" << g->info().param_count;
  }
  os << ",\"shards\":" << cfg_.nodes.size() << "}}]}";
  return os.str();
}

std::string PipelineOrchestrator::cluster_json(bool include_security) const {
  const auto totals = tx_.totals();
  std::ostringstream os;
  os << "{";
  os << "\"cluster\":{\"name\":\"" << json_escape(cfg_.name) << "\",\"self\":" << self_
     << ",\"role\":\"" << (is_master() ? "master" : "worker") << "\""
     << ",\"nodes\":" << cfg_.nodes.size() << ",\"uptime_seconds\":" << uptime_seconds()
     << ",\"requests_served\":" << requests_served_.load()
     << ",\"tokens_generated\":" << tokens_generated_.load() << "}";

  os << ",\"model\":{\"name\":\"" << json_escape(model_name_) << "\",\"path\":\""
     << json_escape(cfg_.model.path) << "\",\"n_layers\":" << cfg_.model.n_layers
     << ",\"hidden_dim\":" << cfg_.model.hidden_dim << ",\"n_vocab\":" << cfg_.model.n_vocab
     << ",\"context_length\":" << cfg_.model.max_seq
     << ",\"weight_bytes\":" << cfg_.model.total_weight_bytes
     << ",\"bytes_per_layer\":" << cfg_.model.bytes_per_layer;
  if (const auto* g = dynamic_cast<const GgufRunner*>(runner_.get())) {
    os << ",\"quantization\":\"" << json_escape(g->info().quantization) << "\""
       << ",\"architecture\":\"" << json_escape(g->info().architecture) << "\""
       << ",\"parameters\":" << g->info().param_count << ",\"bits_per_weight\":"
       << g->info().bits_per_weight;
  }
  os << ",\"runner\":\"" << (runner_ ? runner_->name() : "none") << "\""
     << ",\"compute_backend\":\"" << compute::backend_name() << "\""
     << ",\"compute_threads\":" << compute::thread_count() << "}";

  os << ",\"pipeline\":[";
  for (size_t i = 0; i < dag_.size(); ++i) {
    os << (i ? "," : "") << "{\"node\":" << dag_[i].id << ",\"layers\":{\"start\":"
       << dag_[i].layers.start << ",\"end\":" << dag_[i].layers.end << "}"
       << ",\"embed\":" << (dag_[i].is_embed ? "true" : "false")
       << ",\"lm_head\":" << (dag_[i].is_lm_head ? "true" : "false") << "}";
  }
  os << "]";

  os << ",\"workers\":" << registry_.to_json();

  os << ",\"network\":{\"bytes_sent\":" << totals.bytes_sent << ",\"bytes_recv\":" << totals.bytes_recv
     << ",\"frames_sent\":" << totals.frames_sent << ",\"frames_recv\":" << totals.frames_recv
     << ",\"reconnects\":" << totals.reconnects << ",\"errors\":" << totals.errors
     << ",\"local_frames\":" << totals.local_frames << ",\"peers\":" << totals.peers
     << ",\"connected_peers\":" << totals.connected_peers << ",\"links\":[";
  const auto peers = tx_.peers();
  for (size_t i = 0; i < peers.size(); ++i) {
    os << (i ? "," : "") << "{\"node\":" << peers[i].id << ",\"host\":\"" << json_escape(peers[i].host)
       << "\",\"port\":" << peers[i].port << ",\"connected\":" << (peers[i].connected ? "true" : "false")
       << ",\"bytes_sent\":" << peers[i].bytes_sent << ",\"bytes_recv\":" << peers[i].bytes_recv
       << ",\"frames_sent\":" << peers[i].frames_sent << ",\"frames_recv\":" << peers[i].frames_recv
       << ",\"last_send_ms\":" << peers[i].last_send_ms << ",\"last_recv_ms\":" << peers[i].last_recv_ms
       << ",\"reconnects\":" << peers[i].reconnects << ",\"errors\":" << peers[i].errors << "}";
  }
  os << "]}";

  os << ",\"scheduler\":" << scheduler_.to_json();
  if (include_security) {
    os << ",\"security\":" << security_.status_json();
  } else {
    const auto s = const_cast<security::SecurityGate&>(security_).limiter().total_rejected();
    os << ",\"security\":{\"auth_required\":"
       << (security_.config().require_api_key ? "true" : "false") << ",\"rejected\":" << s << "}";
  }
  os << "}";
  return os.str();
}

std::string PipelineOrchestrator::metrics_text() const {
  const auto s = scheduler_.stats();
  const auto totals = tx_.totals();
  const auto workers = registry_.snapshot();
  size_t alive = 0;
  for (const auto& w : workers) {
    if (w.healthy()) {
      ++alive;
    }
  }
  std::ostringstream os;
  os << "# HELP oracle_requests_total Requests submitted to the scheduler.\n";
  os << "# TYPE oracle_requests_total counter\n";
  os << "oracle_requests_total " << s.submitted << "\n";
  os << "oracle_requests_completed_total " << s.completed << "\n";
  os << "oracle_requests_failed_total " << s.failed << "\n";
  os << "oracle_requests_rejected_total " << s.rejected << "\n";
  os << "oracle_requests_timed_out_total " << s.timed_out << "\n";
  os << "# TYPE oracle_requests_running gauge\n";
  os << "oracle_requests_running " << s.running << "\n";
  os << "oracle_requests_queued " << s.queued << "\n";
  os << "# TYPE oracle_tokens_generated_total counter\n";
  os << "oracle_tokens_generated_total " << s.generated_tokens << "\n";
  os << "# TYPE oracle_tokens_per_second gauge\n";
  os << "oracle_tokens_per_second " << s.tokens_per_second << "\n";
  os << "oracle_queue_wait_ms_avg " << s.avg_queue_ms << "\n";
  os << "oracle_request_duration_ms_avg " << s.avg_run_ms << "\n";
  os << "# TYPE oracle_workers gauge\n";
  os << "oracle_workers_total " << workers.size() << "\n";
  os << "oracle_workers_alive " << alive << "\n";
  os << "# TYPE oracle_network_bytes_total counter\n";
  os << "oracle_network_bytes_sent_total " << totals.bytes_sent << "\n";
  os << "oracle_network_bytes_received_total " << totals.bytes_recv << "\n";
  os << "oracle_network_frames_sent_total " << totals.frames_sent << "\n";
  os << "oracle_network_frames_received_total " << totals.frames_recv << "\n";
  os << "oracle_network_errors_total " << totals.errors << "\n";
  os << "oracle_network_reconnects_total " << totals.reconnects << "\n";
  for (const auto& w : workers) {
    os << "oracle_worker_cpu_load{node=\"" << w.id << "\"} " << w.cpu_load << "\n";
    os << "oracle_worker_ram_free_bytes{node=\"" << w.id << "\"} " << w.ram_free_bytes << "\n";
    os << "oracle_worker_active_requests{node=\"" << w.id << "\"} " << w.active_requests << "\n";
    os << "oracle_worker_healthy{node=\"" << w.id << "\"} " << (w.healthy() ? 1 : 0) << "\n";
  }
  return os.str();
}

Status PipelineOrchestrator::serve_openai(uint16_t port) { return run_openai_server(*this, port); }

}  // namespace oracle

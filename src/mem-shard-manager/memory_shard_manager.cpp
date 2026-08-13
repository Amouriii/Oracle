#include "oracle/shard/memory_shard_manager.hpp"
#include "oracle/tb3/socket_transport.hpp"

#include <algorithm>
#include <cmath>

namespace oracle {

MemoryShardManager::MemoryShardManager(ClusterConfig cfg) : cfg_(std::move(cfg)) {}

uint64_t MemoryShardManager::estimate_layer_bytes(const ModelMeta& m) {
  if (m.bytes_per_layer) {
    return m.bytes_per_layer;
  }
  if (m.total_weight_bytes && m.n_layers) {
    return m.total_weight_bytes / m.n_layers;
  }
  // Q4 70B-class heuristic: ~2*hidden^2 * n_proj * 0.5 bytes, plus FFN 3x.
  const uint64_t h = m.hidden_dim ? m.hidden_dim : 8192;
  const uint64_t attn = 4ull * h * h;       // q,k,v,o
  const uint64_t ffn = 3ull * h * (4ull * h);
  const uint64_t elems = attn + ffn;
  return elems / 2;  // Q4-ish
}

LayerRange MemoryShardManager::layers_for(NodeId id) const {
  if (const auto* n = cfg_.find(id)) {
    return n->layers;
  }
  return {};
}

Status MemoryShardManager::assign_contiguous_layers() {
  if (cfg_.nodes.empty()) {
    return Status::fail(Errc::invalid_argument, "no nodes");
  }
  const uint32_t L = cfg_.model.n_layers;
  if (L == 0) {
    return Status::fail(Errc::invalid_argument, "n_layers == 0");
  }
  const uint32_t n = static_cast<uint32_t>(cfg_.nodes.size());
  uint32_t cursor = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t take = (L - cursor) / (n - i);
    if (take == 0 && cursor < L) {
      take = 1;
    }
    cfg_.nodes[i].layers.start = cursor;
    cfg_.nodes[i].layers.end = std::min(L, cursor + take);
    cursor = cfg_.nodes[i].layers.end;
  }
  if (!cfg_.nodes.empty()) {
    cfg_.nodes.back().layers.end = L;
  }
  return Status::OK();
}

Status MemoryShardManager::plan(ShardPlan* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null plan");
  }
  auto st = assign_contiguous_layers();
  if (!st) {
    return st;
  }
  ShardPlan p;
  p.model = cfg_.model;
  p.nodes = cfg_.nodes;
  const uint64_t layer_b = estimate_layer_bytes(cfg_.model);
  cfg_.model.bytes_per_layer = layer_b;
  p.weight_bytes_total = layer_b * cfg_.model.n_layers;
  p.kv_bytes_total = 0;
  p.feasible = true;
  for (const auto& node : cfg_.nodes) {
    const uint64_t w = layer_b * node.layers.count();
    const KvLayout kv = plan_kv(cfg_.model, node.layers);
    p.kv_bytes_total += kv.bytes_total;
    const uint64_t budget = static_cast<uint64_t>(node.ram_budget_gb * (1ull << 30));
    auto refuse = pressure_.refuse_if_overcommit(w + kv.bytes_total, budget);
    if (!refuse) {
      // Budget check is authoritative for dry-run; host pressure is advisory unless budget fails.
      if (w + kv.bytes_total > budget) {
        p.feasible = false;
        p.reason = "node " + std::to_string(node.id) + " " + refuse.message;
        break;
      }
    }
    if (w + kv.bytes_total > budget) {
      p.feasible = false;
      p.reason = "node " + std::to_string(node.id) + " needs " +
                 std::to_string(w + kv.bytes_total) + " bytes, budget " + std::to_string(budget);
      break;
    }
  }
  if (p.feasible) {
    p.reason = "ok";
  }
  *out = std::move(p);
  return Status::OK();
}

Status MemoryShardManager::stream_layers_init(
    TB3SocketTransport& tx, NodeId self,
    const std::function<Status(NodeId, uint32_t layer, std::span<const std::byte>)>& send_blob) {
  const NodeConfig* me = cfg_.find(self);
  if (!me) {
    return Status::fail(Errc::not_found, "self node missing");
  }
  if (me->role != "master") {
    return Status::OK();
  }
  std::vector<std::byte> blob(static_cast<size_t>(std::min<uint64_t>(estimate_layer_bytes(cfg_.model), 1u << 20)));
  for (const auto& node : cfg_.nodes) {
    if (node.id == self) {
      continue;
    }
    for (uint32_t layer = node.layers.start; layer < node.layers.end; ++layer) {
      TensorHeader hdr;
      hdr.dtype = static_cast<uint16_t>(DType::I8);
      hdr.rank = 1;
      hdr.shape[0] = static_cast<uint32_t>(blob.size());
      hdr.nbytes = blob.size();
      hdr.layer_id = layer;
      hdr.flags = kFlagLayerBlob;
      hdr.checksum = crc32(blob);
      Status st;
      if (send_blob) {
        st = send_blob(node.id, layer, blob);
      } else {
        st = tx.send_tensor(node.id, hdr, blob);
      }
      if (!st) {
        return st;
      }
    }
  }
  return Status::OK();
}

}  // namespace oracle

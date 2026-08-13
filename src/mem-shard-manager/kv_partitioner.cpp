#include "oracle/shard/memory_shard_manager.hpp"

namespace oracle {

KvLayout plan_kv(const ModelMeta& model, LayerRange layers) {
  KvLayout k;
  k.n_local_layers = layers.count();
  k.n_kv_heads = model.n_kv_heads ? model.n_kv_heads : model.n_heads;
  k.max_seq = model.max_seq;
  k.head_dim = model.head_dim ? model.head_dim : (model.n_heads ? model.hidden_dim / model.n_heads : 0);
  k.dtype = model.act_dtype;
  const uint64_t elem = dtype_size(k.dtype);
  k.bytes_k = static_cast<uint64_t>(k.n_local_layers) * k.n_kv_heads * k.max_seq * k.head_dim * elem;
  k.bytes_v = k.bytes_k;
  k.bytes_total = k.bytes_k + k.bytes_v;
  return k;
}

}  // namespace oracle

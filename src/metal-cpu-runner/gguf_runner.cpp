#include "oracle/runner/gguf_runner.hpp"

#include "oracle/compute/blas.hpp"
#include "oracle/model/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace oracle {
namespace {

using model::dequantize_row;
using model::fp16_to_fp32;
using model::fp32_to_fp16;

// Per-thread expansion buffer for one weight row.  Sized on first use and kept
// alive for the process, which keeps the decode loop allocation-free.
std::vector<float>& row_scratch(int64_t n) {
  thread_local std::vector<float> buf;
  if (static_cast<int64_t>(buf.size()) < n) {
    buf.assign(static_cast<size_t>(n), 0.f);
  }
  return buf;
}

void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) {
    ss += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * scale * (w ? w[i] : 1.0f);
  }
}

void softmax_inplace(float* x, int n) {
  if (n <= 0) {
    return;
  }
  float mx = x[0];
  for (int i = 1; i < n; ++i) {
    mx = std::max(mx, x[i]);
  }
  float sum = 0.f;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  const float inv = sum > 0.f ? 1.0f / sum : 0.f;
  for (int i = 0; i < n; ++i) {
    x[i] *= inv;
  }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// ggml's "normal" RoPE: consecutive pairs are rotated together.  GGUF conversion
// permutes Q/K so this reproduces the reference HF rotation.
void rope(float* vec, int n_heads, int head_dim, int rope_dim, uint32_t pos, float freq_base,
          float freq_scale) {
  const int rot = std::min(rope_dim, head_dim);
  for (int h = 0; h < n_heads; ++h) {
    float* v = vec + static_cast<size_t>(h) * head_dim;
    for (int i = 0; i < rot; i += 2) {
      const float freq = std::pow(freq_base, -static_cast<float>(i) / static_cast<float>(rot));
      const float theta = static_cast<float>(pos) * freq * freq_scale;
      const float c = std::cos(theta);
      const float s = std::sin(theta);
      const float x0 = v[i];
      const float x1 = v[i + 1];
      v[i] = x0 * c - x1 * s;
      v[i + 1] = x0 * s + x1 * c;
    }
  }
}

Status dequant_vector(const model::GgufFile& f, const model::GgufTensorInfo* t, std::vector<float>* out,
                      const char* what) {
  if (!t) {
    out->clear();
    return Status::OK();
  }
  const uint8_t* p = f.tensor_data(*t);
  if (!p) {
    return Status::fail(Errc::protocol, std::string(what) + ": tensor data out of range");
  }
  out->assign(static_cast<size_t>(t->n_elements), 0.f);
  return dequantize_row(t->type, p, out->data(), static_cast<int64_t>(t->n_elements));
}

Status bind_weight(const model::GgufFile& f, const std::string& name, WeightView* out, bool required,
            const char* what) {
  const auto* t = f.find_tensor(name);
  if (!t) {
    if (required) {
      return Status::fail(Errc::not_found, std::string(what) + ": missing tensor " + name);
    }
    *out = WeightView{};
    return Status::OK();
  }
  const uint8_t* p = f.tensor_data(*t);
  if (!p) {
    return Status::fail(Errc::protocol, name + ": tensor data extends past end of file");
  }
  if (t->ne.size() < 2) {
    return Status::fail(Errc::protocol, name + ": expected a 2-D weight matrix");
  }
  if (!model::dequantize_supported(t->type)) {
    return Status::fail(Errc::not_implemented,
                        name + ": no dequantiser for " + model::ggml_type_info(t->type).name);
  }
  out->data = p;
  out->type = t->type;
  out->cols = static_cast<int64_t>(t->ne[0]);
  out->rows = static_cast<int64_t>(t->rows());
  out->row_bytes = static_cast<int64_t>(model::ggml_row_bytes(t->type, t->ne[0]));
  return Status::OK();
}

std::string blk(uint32_t i, const char* suffix) {
  return "blk." + std::to_string(i) + "." + suffix;
}

}  // namespace

Status weight_matvec(const WeightView& w, const float* x, float* y) {
  if (!w.valid() || !x || !y) {
    return Status::fail(Errc::invalid_argument, "matvec: invalid weight view");
  }
  if (w.type == model::GGML_F32) {
    compute::parallel_for(static_cast<int>(w.rows), [&](int lo, int hi) {
      for (int r = lo; r < hi; ++r) {
        const auto* row = reinterpret_cast<const float*>(w.data + static_cast<size_t>(r) * w.row_bytes);
        y[r] = compute::dot(row, x, static_cast<int>(w.cols));
      }
    });
    return Status::OK();
  }
  if (w.type == model::GGML_F16) {
    compute::parallel_for(static_cast<int>(w.rows), [&](int lo, int hi) {
      for (int r = lo; r < hi; ++r) {
        const auto* row = reinterpret_cast<const uint16_t*>(w.data + static_cast<size_t>(r) * w.row_bytes);
        y[r] = compute::dot_f16(row, x, static_cast<int>(w.cols));
      }
    });
    return Status::OK();
  }

  Status first = Status::OK();
  compute::parallel_for(static_cast<int>(w.rows), [&](int lo, int hi) {
    std::vector<float>& buf = row_scratch(w.cols);
    for (int r = lo; r < hi; ++r) {
      auto st = dequantize_row(w.type, w.data + static_cast<size_t>(r) * w.row_bytes, buf.data(), w.cols);
      if (!st) {
        if (first.ok()) {
          first = st;
        }
        return;
      }
      y[r] = compute::dot(buf.data(), x, static_cast<int>(w.cols));
    }
  });
  return first;
}

GgufRunner::GgufRunner() = default;
GgufRunner::~GgufRunner() = default;

uint64_t GgufRunner::kv_bytes(uint32_t max_seq, DType dtype) const {
  return 2ull * layers_.count() * info_.n_kv_heads * info_.head_dim * max_seq * dtype_size(dtype);
}

Status GgufRunner::load_layers(const ModelMeta& model, LayerRange layers, std::string_view weights_path) {
  std::string path(weights_path.empty() ? model.path : std::string(weights_path));
  if (path.empty()) {
    return Status::fail(Errc::invalid_argument,
                        "the gguf runner needs a model path (set [model] path in the cluster config "
                        "or pass --model)");
  }
  auto st = file_.open(path);
  if (!st) {
    return st;
  }
  st = model::inspect_gguf(file_, &info_);
  if (!st) {
    return st;
  }
  if (!info_.supported_for_inference) {
    return Status::fail(Errc::not_implemented, info_.unsupported_reason);
  }

  meta_ = model;
  model::apply_to_model_meta(info_, &meta_);
  if (model.max_seq && model.max_seq < meta_.max_seq) {
    meta_.max_seq = model.max_seq;  // config caps context to keep the KV cache affordable
  }
  meta_.act_dtype = model.act_dtype;

  layers_ = layers;
  if (layers_.end == 0 && layers_.start == 0) {
    layers_ = LayerRange{0, info_.n_layers};
  }
  if (layers_.end > info_.n_layers) {
    layers_.end = info_.n_layers;
  }
  if (layers_.start > layers_.end) {
    return Status::fail(Errc::invalid_argument, "layer range start is past its end");
  }

  auto tok_st = tokenizer_.load(file_);
  (void)tok_st;  // a missing vocab degrades to byte tokens rather than failing the load

  return bind_weights();
}

Status GgufRunner::bind_weights() {
  blocks_.assign(layers_.count(), GgufLayer{});
  resident_bytes_ = 0;

  const bool first_stage = layers_.start == 0;
  const bool last_stage = layers_.end == info_.n_layers;

  if (first_stage) {
    auto st = bind_weight(file_, "token_embd.weight", &embed_, true, "embedding");
    if (!st) {
      return st;
    }
    resident_bytes_ += static_cast<uint64_t>(embed_.rows) * static_cast<uint64_t>(embed_.row_bytes);
  }

  for (uint32_t li = layers_.start; li < layers_.end; ++li) {
    GgufLayer& L = blocks_[li - layers_.start];
    auto st = dequant_vector(file_, file_.find_tensor(blk(li, "attn_norm.weight")), &L.attn_norm,
                             "attn_norm");
    if (!st) {
      return st;
    }
    st = dequant_vector(file_, file_.find_tensor(blk(li, "ffn_norm.weight")), &L.ffn_norm, "ffn_norm");
    if (!st) {
      return st;
    }
    struct Bind {
      const char* suffix;
      WeightView* dst;
    };
    const Bind binds[] = {
        {"attn_q.weight", &L.wq},   {"attn_k.weight", &L.wk},     {"attn_v.weight", &L.wv},
        {"attn_output.weight", &L.wo}, {"ffn_gate.weight", &L.w_gate}, {"ffn_up.weight", &L.w_up},
        {"ffn_down.weight", &L.w_down},
    };
    for (const auto& b : binds) {
      st = bind_weight(file_, blk(li, b.suffix), b.dst, true, "layer");
      if (!st) {
        return st;
      }
      resident_bytes_ += static_cast<uint64_t>(b.dst->rows) * static_cast<uint64_t>(b.dst->row_bytes);
    }
    // Optional projection biases (qwen2 and friends).
    const std::pair<const char*, std::vector<float>*> bias_binds[] = {
        {"attn_q.bias", &L.bq}, {"attn_k.bias", &L.bk}, {"attn_v.bias", &L.bv},
        {"attn_output.bias", &L.bo},
    };
    for (const auto& [suffix, dst] : bias_binds) {
      st = dequant_vector(file_, file_.find_tensor(blk(li, suffix)), dst, "bias");
      if (!st) {
        return st;
      }
    }
    L.loaded = true;
  }

  if (last_stage) {
    auto st = dequant_vector(file_, file_.find_tensor("output_norm.weight"), &output_norm_, "output_norm");
    if (!st) {
      return st;
    }
    st = bind_weight(file_, "output.weight", &output_, false, "lm_head");
    if (!st) {
      return st;
    }
    if (!output_.valid()) {
      // Tied embeddings: the output head reuses token_embd.
      st = bind_weight(file_, "token_embd.weight", &output_, true, "lm_head (tied)");
      if (!st) {
        return st;
      }
    }
    resident_bytes_ += static_cast<uint64_t>(output_.rows) * static_cast<uint64_t>(output_.row_bytes);
  }

  return Status::OK();
}

void GgufScratch::resize(uint32_t n_embd, uint32_t n_ff, uint32_t q_dim, uint32_t kv_dim) {
  x.assign(n_embd, 0.f);
  xb.assign(n_embd, 0.f);
  xb2.assign(std::max(n_embd, q_dim), 0.f);
  q.assign(q_dim, 0.f);
  k.assign(kv_dim, 0.f);
  v.assign(kv_dim, 0.f);
  attn_out.assign(q_dim, 0.f);
  hb.assign(std::max(n_ff, 1u), 0.f);
  hb2.assign(std::max(n_ff, 1u), 0.f);
}

Status GgufRunner::load_layer_blob(uint32_t, std::span<const std::byte>) {
  // Weights are read from each node's own copy of the GGUF file rather than
  // streamed over the wire, so there is nothing to install here.
  return Status::OK();
}

Status GgufRunner::embed(std::span<const int32_t> tokens, Tensor* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "embed: null out");
  }
  if (!embed_.valid()) {
    return Status::fail(Errc::invalid_argument, "embed: this node does not own token_embd");
  }
  const int64_t h = embed_.cols;
  std::vector<float> acc(static_cast<size_t>(h) * tokens.size(), 0.f);
  std::vector<float>& buf = row_scratch(h);
  for (size_t i = 0; i < tokens.size(); ++i) {
    int32_t tok = tokens[i];
    if (tok < 0 || tok >= embed_.rows) {
      tok = 0;
    }
    auto st = dequantize_row(embed_.type, embed_.data + static_cast<size_t>(tok) * embed_.row_bytes,
                             buf.data(), h);
    if (!st) {
      return st;
    }
    std::memcpy(acc.data() + static_cast<size_t>(i) * h, buf.data(), static_cast<size_t>(h) * 4);
  }

  out->header = TensorHeader{};
  out->header.dtype = static_cast<uint16_t>(meta_.act_dtype);
  out->header.rank = 2;
  out->header.shape[0] = static_cast<uint32_t>(tokens.size());
  out->header.shape[1] = static_cast<uint32_t>(h);
  if (meta_.act_dtype == DType::F16) {
    out->payload.resize(acc.size() * 2);
    model::fp32_to_fp16_row(acc.data(), reinterpret_cast<uint16_t*>(out->payload.data()),
                            static_cast<int64_t>(acc.size()));
  } else {
    out->payload.resize(acc.size() * 4);
    std::memcpy(out->payload.data(), acc.data(), acc.size() * 4);
  }
  out->header.nbytes = out->payload.size();
  out->header.checksum = crc32(out->payload);
  return Status::OK();
}

Status GgufRunner::forward_token(const float* in, float* out, KvCache& kv, uint32_t pos,
                                 GgufScratch& s) const {
  const int n_embd = static_cast<int>(info_.n_embd);
  const int n_heads = static_cast<int>(info_.n_heads);
  const int n_kv_heads = static_cast<int>(info_.n_kv_heads);
  const int head_dim = static_cast<int>(info_.head_dim);
  const int kv_dim = n_kv_heads * head_dim;
  const int gqa = n_heads / std::max(1, n_kv_heads);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const uint32_t max_seq = kv.layout.max_seq ? kv.layout.max_seq : meta_.max_seq;
  if (pos >= max_seq) {
    return Status::fail(Errc::invalid_argument,
                        "context window exhausted at " + std::to_string(max_seq) + " tokens");
  }
  const bool kv_f16 = kv.layout.dtype == DType::F16;
  const size_t kv_elem = kv_f16 ? 2 : 4;
  const size_t layer_stride = static_cast<size_t>(max_seq) * static_cast<size_t>(kv_dim) * kv_elem;
  if (kv.k.size() < layer_stride * blocks_.size() || kv.v.size() < layer_stride * blocks_.size()) {
    return Status::fail(Errc::oom, "kv cache is smaller than this node's layer shard needs");
  }

  std::copy(in, in + n_embd, s.x.begin());

  for (size_t li = 0; li < blocks_.size(); ++li) {
    const GgufLayer& L = blocks_[li];
    rmsnorm(s.x.data(), L.attn_norm.empty() ? nullptr : L.attn_norm.data(), s.xb.data(), n_embd,
            info_.rms_eps);

    auto st = weight_matvec(L.wq, s.xb.data(), s.q.data());
    if (!st) return st;
    st = weight_matvec(L.wk, s.xb.data(), s.k.data());
    if (!st) return st;
    st = weight_matvec(L.wv, s.xb.data(), s.v.data());
    if (!st) return st;
    for (size_t i = 0; i < L.bq.size() && i < s.q.size(); ++i) s.q[i] += L.bq[i];
    for (size_t i = 0; i < L.bk.size() && i < s.k.size(); ++i) s.k[i] += L.bk[i];
    for (size_t i = 0; i < L.bv.size() && i < s.v.size(); ++i) s.v[i] += L.bv[i];

    rope(s.q.data(), n_heads, head_dim, static_cast<int>(info_.rope_dim), pos, info_.rope_freq_base,
         info_.rope_freq_scale);
    rope(s.k.data(), n_kv_heads, head_dim, static_cast<int>(info_.rope_dim), pos, info_.rope_freq_base,
         info_.rope_freq_scale);

    std::byte* kbase = kv.k.data() + layer_stride * li;
    std::byte* vbase = kv.v.data() + layer_stride * li;
    const size_t row_off = static_cast<size_t>(pos) * static_cast<size_t>(kv_dim) * kv_elem;
    if (kv_f16) {
      model::fp32_to_fp16_row(s.k.data(), reinterpret_cast<uint16_t*>(kbase + row_off), kv_dim);
      model::fp32_to_fp16_row(s.v.data(), reinterpret_cast<uint16_t*>(vbase + row_off), kv_dim);
    } else {
      std::memcpy(kbase + row_off, s.k.data(), static_cast<size_t>(kv_dim) * 4);
      std::memcpy(vbase + row_off, s.v.data(), static_cast<size_t>(kv_dim) * 4);
    }

    // Grouped-query attention over positions [0, pos].
    const int n_ctx = static_cast<int>(pos) + 1;
    compute::parallel_for(n_heads, [&](int lo, int hi) {
      std::vector<float> scores(static_cast<size_t>(n_ctx));
      for (int h = lo; h < hi; ++h) {
        const int kvh = h / gqa;
        const float* qh = s.q.data() + static_cast<size_t>(h) * head_dim;
        for (int t = 0; t < n_ctx; ++t) {
          const size_t off = (static_cast<size_t>(t) * kv_dim + static_cast<size_t>(kvh) * head_dim) * kv_elem;
          scores[static_cast<size_t>(t)] =
              kv_f16 ? compute::dot_f16(reinterpret_cast<const uint16_t*>(kbase + off), qh, head_dim) * scale
                     : compute::dot(reinterpret_cast<const float*>(kbase + off), qh, head_dim) * scale;
        }
        softmax_inplace(scores.data(), n_ctx);
        float* oh = s.attn_out.data() + static_cast<size_t>(h) * head_dim;
        std::fill(oh, oh + head_dim, 0.f);
        for (int t = 0; t < n_ctx; ++t) {
          const float w = scores[static_cast<size_t>(t)];
          if (w == 0.f) {
            continue;
          }
          const size_t off = (static_cast<size_t>(t) * kv_dim + static_cast<size_t>(kvh) * head_dim) * kv_elem;
          if (kv_f16) {
            const auto* vp = reinterpret_cast<const uint16_t*>(vbase + off);
            for (int d = 0; d < head_dim; ++d) {
              oh[d] += w * fp16_to_fp32(vp[d]);
            }
          } else {
            const auto* vp = reinterpret_cast<const float*>(vbase + off);
            for (int d = 0; d < head_dim; ++d) {
              oh[d] += w * vp[d];
            }
          }
        }
      }
    });

    st = weight_matvec(L.wo, s.attn_out.data(), s.xb2.data());
    if (!st) return st;
    for (size_t i = 0; i < L.bo.size() && i < s.xb2.size(); ++i) s.xb2[i] += L.bo[i];
    for (int i = 0; i < n_embd; ++i) {
      s.x[static_cast<size_t>(i)] += s.xb2[static_cast<size_t>(i)];
    }

    // SwiGLU feed-forward.
    rmsnorm(s.x.data(), L.ffn_norm.empty() ? nullptr : L.ffn_norm.data(), s.xb.data(), n_embd,
            info_.rms_eps);
    st = weight_matvec(L.w_gate, s.xb.data(), s.hb.data());
    if (!st) return st;
    st = weight_matvec(L.w_up, s.xb.data(), s.hb2.data());
    if (!st) return st;
    const int n_ff = static_cast<int>(L.w_gate.rows);
    for (int i = 0; i < n_ff; ++i) {
      s.hb[static_cast<size_t>(i)] = silu(s.hb[static_cast<size_t>(i)]) * s.hb2[static_cast<size_t>(i)];
    }
    st = weight_matvec(L.w_down, s.hb.data(), s.xb2.data());
    if (!st) return st;
    for (int i = 0; i < n_embd; ++i) {
      s.x[static_cast<size_t>(i)] += s.xb2[static_cast<size_t>(i)];
    }
  }

  std::copy(s.x.begin(), s.x.begin() + n_embd, out);
  return Status::OK();
}

Status GgufRunner::forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "forward: null out");
  }
  const int n_embd = static_cast<int>(info_.n_embd);
  if (n_embd <= 0) {
    return Status::fail(Errc::invalid_argument, "forward: model not loaded");
  }
  // Infer the wire dtype from the byte count, preferring the configured
  // activation dtype, so a peer that still speaks f32 keeps working.
  const size_t row_f32 = static_cast<size_t>(n_embd) * 4;
  const size_t row_f16 = static_cast<size_t>(n_embd) * 2;
  const bool fits_f32 = in.size() % row_f32 == 0;
  const bool fits_f16 = in.size() % row_f16 == 0;
  size_t n_tokens = 0;
  bool payload_is_f16 = false;
  if (meta_.act_dtype == DType::F16 && fits_f16) {
    n_tokens = in.size() / row_f16;
    payload_is_f16 = true;
  } else if (fits_f32) {
    n_tokens = in.size() / row_f32;
  } else if (fits_f16) {
    n_tokens = in.size() / row_f16;
    payload_is_f16 = true;
  }
  if (n_tokens == 0) {
    return Status::fail(Errc::invalid_argument,
                        "forward: payload of " + std::to_string(in.size()) +
                            " bytes is not a whole number of " + std::to_string(n_embd) +
                            "-wide activations");
  }

  std::vector<float> hidden(static_cast<size_t>(n_embd) * n_tokens);
  if (payload_is_f16) {
    model::fp16_to_fp32_row(reinterpret_cast<const uint16_t*>(in.data()), hidden.data(),
                            static_cast<int64_t>(hidden.size()));
  } else {
    std::memcpy(hidden.data(), in.data(), hidden.size() * 4);
  }

  GgufScratch scratch;
  scratch.resize(info_.n_embd, info_.n_ff, info_.n_heads * info_.head_dim,
                 info_.n_kv_heads * info_.head_dim);
  std::vector<float> result(hidden.size());
  for (size_t t = 0; t < n_tokens; ++t) {
    auto st = forward_token(hidden.data() + t * static_cast<size_t>(n_embd),
                            result.data() + t * static_cast<size_t>(n_embd), kv, kv.seq_len, scratch);
    if (!st) {
      return st;
    }
    ++kv.seq_len;
  }

  out->header = TensorHeader{};
  out->header.dtype = static_cast<uint16_t>(meta_.act_dtype);
  out->header.rank = 2;
  out->header.shape[0] = static_cast<uint32_t>(n_tokens);
  out->header.shape[1] = static_cast<uint32_t>(n_embd);
  out->header.flags = kFlagDecode;
  if (meta_.act_dtype == DType::F16) {
    out->payload.resize(result.size() * 2);
    model::fp32_to_fp16_row(result.data(), reinterpret_cast<uint16_t*>(out->payload.data()),
                            static_cast<int64_t>(result.size()));
  } else {
    out->payload.resize(result.size() * 4);
    std::memcpy(out->payload.data(), result.data(), result.size() * 4);
  }
  out->header.nbytes = out->payload.size();
  out->header.checksum = crc32(out->payload);
  return Status::OK();
}

Status GgufRunner::lm_head(std::span<const std::byte> hidden, Tensor* logits) {
  if (!logits) {
    return Status::fail(Errc::invalid_argument, "lm_head: null out");
  }
  if (!output_.valid()) {
    return Status::fail(Errc::invalid_argument, "lm_head: this node does not own the output head");
  }
  const size_t n_embd = info_.n_embd;
  const bool f16 = meta_.act_dtype == DType::F16;
  const size_t stride = n_embd * (f16 ? 2 : 4);
  if (stride == 0 || hidden.size() < stride) {
    return Status::fail(Errc::invalid_argument, "lm_head: hidden state is too small");
  }
  // Only the final position produces the next-token distribution.
  const std::byte* last = hidden.data() + (hidden.size() / stride - 1) * stride;
  std::vector<float> x(n_embd);
  if (f16) {
    model::fp16_to_fp32_row(reinterpret_cast<const uint16_t*>(last), x.data(),
                            static_cast<int64_t>(n_embd));
  } else {
    std::memcpy(x.data(), last, n_embd * 4);
  }

  std::vector<float> normed(n_embd);
  rmsnorm(x.data(), output_norm_.empty() ? nullptr : output_norm_.data(), normed.data(),
          static_cast<int>(n_embd), info_.rms_eps);

  std::vector<float> out(static_cast<size_t>(output_.rows));
  auto st = weight_matvec(output_, normed.data(), out.data());
  if (!st) {
    return st;
  }

  logits->header = TensorHeader{};
  logits->header.dtype = static_cast<uint16_t>(DType::F32);
  logits->header.rank = 1;
  logits->header.shape[0] = static_cast<uint32_t>(out.size());
  logits->header.flags = kFlagLastStageLogits;
  logits->payload.resize(out.size() * 4);
  std::memcpy(logits->payload.data(), out.data(), out.size() * 4);
  logits->header.nbytes = logits->payload.size();
  logits->header.checksum = crc32(logits->payload);
  return Status::OK();
}

std::unique_ptr<NodeRunner> make_gguf_runner() { return std::make_unique<GgufRunner>(); }

}  // namespace oracle

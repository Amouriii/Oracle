#include "Accelerate/Accelerate.h"

#include "oracle/runner/node_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace oracle {
namespace {

void f16_to_f32(std::span<const std::byte> in, std::vector<float>& out) {
  const size_t n = in.size() / 2;
  out.resize(n);
  const auto* src = reinterpret_cast<const uint16_t*>(in.data());
  for (size_t i = 0; i < n; ++i) {
    const uint16_t h = src[i];
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
      f = (sign << 31);
    } else if (exp == 31) {
      f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
      f = (sign << 31) | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    std::memcpy(&out[i], &f, 4);
  }
}

void f32_to_f16(std::span<const float> in, std::vector<std::byte>& out) {
  out.resize(in.size() * 2);
  auto* dst = reinterpret_cast<uint16_t*>(out.data());
  for (size_t i = 0; i < in.size(); ++i) {
    uint32_t f;
    std::memcpy(&f, &in[i], 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = f & 0x7FFFFFu;
    uint16_t h;
    if (exp <= 0) {
      h = static_cast<uint16_t>(sign);
    } else if (exp >= 31) {
      h = static_cast<uint16_t>(sign | 0x7C00u);
    } else {
      h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
    }
    dst[i] = h;
  }
}

void identity_square(std::vector<float>& w, uint32_t n) {
  w.assign(static_cast<size_t>(n) * n, 0.f);
  for (uint32_t i = 0; i < n; ++i) {
    w[i * n + i] = 1.f;
  }
}

}  // namespace

Status KvCache::allocate(const KvLayout& l) {
  layout = l;
  seq_len = 0;
  try {
    k.assign(static_cast<size_t>(l.bytes_k), std::byte{0});
    v.assign(static_cast<size_t>(l.bytes_v), std::byte{0});
  } catch (...) {
    return Status::fail(Errc::oom, "kv allocate");
  }
  return Status::OK();
}

void KvCache::reset() {
  seq_len = 0;
  std::fill(k.begin(), k.end(), std::byte{0});
  std::fill(v.begin(), v.end(), std::byte{0});
}

Status AccelerateRunner::load_layers(const ModelMeta& model, LayerRange layers, std::string_view) {
  model_ = model;
  layers_ = layers;
  const uint32_t h = model.hidden_dim;
  const uint32_t n = layers.count();
  w_in_.resize(n);
  w_out_.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    identity_square(w_in_[i], h);
    identity_square(w_out_[i], h);
  }
  embed_w_.assign(static_cast<size_t>(model.n_vocab) * h, 0.f);
  for (uint32_t t = 0; t < model.n_vocab && t < h; ++t) {
    embed_w_[static_cast<size_t>(t) * h + (t % h)] = 1.f;
  }
  lm_w_.assign(static_cast<size_t>(model.n_vocab) * h, 0.f);
  for (uint32_t t = 0; t < model.n_vocab && t < h; ++t) {
    lm_w_[static_cast<size_t>(t) * h + (t % h)] = 1.f;
  }
  return Status::OK();
}

Status AccelerateRunner::load_layer_blob(uint32_t layer, std::span<const std::byte> blob) {
  if (layer < layers_.start || layer >= layers_.end) {
    return Status::fail(Errc::invalid_argument, "layer out of range");
  }
  (void)blob;
  return Status::OK();
}

Status AccelerateRunner::gemm_f32(uint32_t m, uint32_t n, uint32_t k, const float* a, const float* b, float* c,
                                  float alpha, float beta) {
  if (!a || !b || !c || m == 0 || n == 0 || k == 0) {
    return Status::fail(Errc::invalid_argument, "sgemm args");
  }
  // C (m x n) = A (m x k) * B (k x n), row-major via cblas
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), alpha, a, static_cast<int>(k), b, static_cast<int>(n), beta, c, static_cast<int>(n));
  return Status::OK();
}

Status AccelerateRunner::embed(std::span<const int32_t> tokens, Tensor* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null out");
  }
  const uint32_t h = model_.hidden_dim;
  std::vector<float> acc(h, 0.f);
  for (auto tok : tokens) {
    if (tok < 0 || static_cast<uint32_t>(tok) >= model_.n_vocab) {
      continue;
    }
    const float* row = embed_w_.data() + static_cast<size_t>(tok) * h;
    for (uint32_t i = 0; i < h; ++i) {
      acc[i] += row[i];
    }
  }
  if (!tokens.empty()) {
    const float inv = 1.f / static_cast<float>(tokens.size());
    for (auto& x : acc) {
      x *= inv;
    }
  }
  out->header.magic = kTensorMagic;
  out->header.dtype = static_cast<uint16_t>(model_.act_dtype);
  out->header.rank = 1;
  out->header.shape[0] = h;
  if (model_.act_dtype == DType::F16) {
    f32_to_f16(acc, out->payload);
  } else {
    out->payload.resize(acc.size() * 4);
    std::memcpy(out->payload.data(), acc.data(), acc.size() * 4);
  }
  out->header.nbytes = out->payload.size();
  return Status::OK();
}

Status AccelerateRunner::forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null out");
  }
  const uint32_t h = model_.hidden_dim;
  std::vector<float> x;
  if (in.size() == h * 4) {
    x.resize(h);
    std::memcpy(x.data(), in.data(), h * 4);
  } else {
    f16_to_f32(in, x);
  }
  if (x.size() < h) {
    x.resize(h, 0.f);
  }
  std::vector<float> tmp(h), y(h);
  for (size_t li = 0; li < w_in_.size(); ++li) {
    auto st = gemm_f32(1, h, h, x.data(), w_in_[li].data(), tmp.data());
    if (!st) {
      return st;
    }
    st = gemm_f32(1, h, h, tmp.data(), w_out_[li].data(), y.data());
    if (!st) {
      return st;
    }
    x.swap(y);
  }
  if (kv.seq_len + 1 <= kv.layout.max_seq) {
    ++kv.seq_len;
  }
  out->header.magic = kTensorMagic;
  out->header.dtype = static_cast<uint16_t>(model_.act_dtype);
  out->header.rank = 1;
  out->header.shape[0] = h;
  out->header.flags = kFlagDecode;
  if (model_.act_dtype == DType::F16) {
    f32_to_f16(x, out->payload);
  } else {
    out->payload.resize(x.size() * 4);
    std::memcpy(out->payload.data(), x.data(), x.size() * 4);
  }
  out->header.nbytes = out->payload.size();
  out->header.checksum = crc32(out->payload);
  return Status::OK();
}

Status AccelerateRunner::lm_head(std::span<const std::byte> hidden, Tensor* logits) {
  if (!logits) {
    return Status::fail(Errc::invalid_argument, "null logits");
  }
  const uint32_t h = model_.hidden_dim;
  const uint32_t v = model_.n_vocab;
  std::vector<float> x;
  if (hidden.size() == h * 4) {
    x.resize(h);
    std::memcpy(x.data(), hidden.data(), h * 4);
  } else {
    f16_to_f32(hidden, x);
  }
  std::vector<float> out(v, 0.f);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, static_cast<int>(v), static_cast<int>(h), 1.f,
              x.data(), static_cast<int>(h), lm_w_.data(), static_cast<int>(h), 0.f, out.data(),
              static_cast<int>(v));
  Status st = Status::OK();
  if (!st) {
    return st;
  }
  logits->header.magic = kTensorMagic;
  logits->header.dtype = static_cast<uint16_t>(DType::F32);
  logits->header.rank = 1;
  logits->header.shape[0] = v;
  logits->header.nbytes = out.size() * 4;
  logits->header.flags = kFlagLastStageLogits;
  logits->payload.resize(out.size() * 4);
  std::memcpy(logits->payload.data(), out.data(), out.size() * 4);
  return Status::OK();
}

std::unique_ptr<NodeRunner> make_accelerate_runner() { return std::make_unique<AccelerateRunner>(); }

}  // namespace oracle

// Loads a synthetic llama-architecture GGUF and checks that:
//   1. recognition reports the right geometry, quantisation and memory needs,
//   2. a single node produces finite logits,
//   3. splitting the layers across two nodes produces the *same* logits, which
//      is the property the whole distributed pipeline rests on.
#include "gguf_fixture.hpp"

#include "oracle/model/gguf.hpp"
#include "oracle/runner/gguf_runner.hpp"

#include <algorithm>
#include "check.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

oracle::KvLayout layout_for(const oracle_test::TinyModel& m, uint32_t n_layers, oracle::DType dt) {
  oracle::KvLayout l;
  l.n_local_layers = n_layers;
  l.n_kv_heads = m.n_kv_heads;
  l.max_seq = m.context_length;
  l.head_dim = m.head_dim;
  l.dtype = dt;
  l.bytes_k = static_cast<uint64_t>(n_layers) * m.n_kv_heads * m.head_dim * m.context_length *
              oracle::dtype_size(dt);
  l.bytes_v = l.bytes_k;
  l.bytes_total = l.bytes_k + l.bytes_v;
  return l;
}

std::vector<float> logits_of(const oracle::Tensor& t) {
  std::vector<float> v(t.payload.size() / 4);
  std::memcpy(v.data(), t.payload.data(), v.size() * 4);
  return v;
}

}  // namespace

int main() {
  const auto path = (std::filesystem::temp_directory_path() / "oracle-tiny.gguf").string();
  const auto m = oracle_test::build_tiny_gguf(path);
  if (m.n_vocab == 0) {
    std::cerr << "could not write fixture to " << path << "\n";
    return 1;
  }

  // ---- recognition -------------------------------------------------------
  oracle::model::ModelInfo info;
  auto st = oracle::model::inspect_gguf(path, &info);
  if (!st) {
    std::cerr << "inspect: " << st.message << "\n";
    return 1;
  }
  CHECK(info.architecture == "llama");
  CHECK(info.name == "oracle-tiny-test");
  CHECK(info.n_layers == m.n_layers);
  CHECK(info.n_embd == m.n_embd);
  CHECK(info.n_heads == m.n_heads);
  CHECK(info.n_kv_heads == m.n_kv_heads);
  CHECK(info.head_dim == m.head_dim);
  CHECK(info.n_ff == m.n_ff);
  CHECK(info.n_vocab == m.n_vocab);
  CHECK(info.context_length == m.context_length);
  CHECK(info.quantization == "F32");
  CHECK(info.dominant_type == "F32");
  CHECK(info.supported_for_inference);
  CHECK(info.param_count > 0);
  CHECK(std::abs(info.bits_per_weight - 32.0) < 1e-6);
  CHECK(info.weight_bytes_total == info.param_count * 4);
  CHECK(info.bytes_per_layer > 0);
  CHECK(info.kv_bytes_per_token == 2ull * m.n_layers * m.n_kv_heads * m.head_dim * 2);
  CHECK(info.recommended_ram_bytes > info.weight_bytes_total);
  std::cout << "recognised " << info.param_count_human() << " params, " << info.quantization << ", "
            << info.n_layers << " layers\n";

  // A malformed path must fail cleanly rather than crash.
  oracle::model::ModelInfo bad;
  CHECK(!oracle::model::inspect_gguf("/nonexistent/oracle.gguf", &bad).ok());

  // ---- single-node inference --------------------------------------------
  oracle::ModelMeta meta;
  meta.path = path;
  meta.act_dtype = oracle::DType::F32;
  oracle::GgufRunner whole;
  st = whole.load_layers(meta, {0, m.n_layers}, path);
  if (!st) {
    std::cerr << "load: " << st.message << "\n";
    return 1;
  }
  CHECK(whole.owns_embedding());
  CHECK(whole.owns_output_head());
  CHECK(whole.resident_weight_bytes() > 0);

  const std::vector<int32_t> prompt{1, 260, 261, 262};
  oracle::KvCache kv_whole;
  st = kv_whole.allocate(layout_for(m, m.n_layers, oracle::DType::F32));
  CHECK(st.ok());

  oracle::Tensor embedded;
  st = whole.embed(prompt, &embedded);
  if (!st) {
    std::cerr << "embed: " << st.message << "\n";
    return 1;
  }
  CHECK(embedded.header.shape[0] == prompt.size());
  CHECK(embedded.header.shape[1] == m.n_embd);

  oracle::Tensor hidden;
  st = whole.forward(embedded.payload, kv_whole, &hidden);
  if (!st) {
    std::cerr << "forward: " << st.message << "\n";
    return 1;
  }
  CHECK(kv_whole.seq_len == prompt.size());

  oracle::Tensor logits;
  st = whole.lm_head(hidden.payload, &logits);
  if (!st) {
    std::cerr << "lm_head: " << st.message << "\n";
    return 1;
  }
  const auto single = logits_of(logits);
  CHECK(single.size() == m.n_vocab);
  for (float v : single) {
    CHECK(std::isfinite(v));
  }

  // ---- the same model split across two nodes -----------------------------
  const uint32_t split = m.n_layers / 2;
  oracle::GgufRunner front, back;
  st = front.load_layers(meta, {0, split}, path);
  CHECK(st.ok());
  st = back.load_layers(meta, {split, m.n_layers}, path);
  CHECK(st.ok());
  CHECK(front.owns_embedding());
  CHECK(!front.owns_output_head());
  CHECK(!back.owns_embedding());
  CHECK(back.owns_output_head());

  oracle::KvCache kv_front, kv_back;
  CHECK(kv_front.allocate(layout_for(m, split, oracle::DType::F32)).ok());
  CHECK(kv_back.allocate(layout_for(m, m.n_layers - split, oracle::DType::F32)).ok());

  oracle::Tensor e2, h1, h2, l2;
  CHECK(front.embed(prompt, &e2).ok());
  st = front.forward(e2.payload, kv_front, &h1);
  if (!st) {
    std::cerr << "front forward: " << st.message << "\n";
    return 1;
  }
  st = back.forward(h1.payload, kv_back, &h2);
  if (!st) {
    std::cerr << "back forward: " << st.message << "\n";
    return 1;
  }
  st = back.lm_head(h2.payload, &l2);
  CHECK(st.ok());
  const auto sharded = logits_of(l2);
  CHECK(sharded.size() == single.size());

  double max_diff = 0.0;
  for (size_t i = 0; i < single.size(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::abs(single[i] - sharded[i])));
  }
  std::cout << "pipeline split " << split << "/" << m.n_layers << " max logit delta " << max_diff << "\n";
  CHECK(max_diff < 1e-4);

  // ---- decode step continues the KV cache --------------------------------
  const int32_t next = static_cast<int32_t>(std::max_element(single.begin(), single.end()) -
                                            single.begin());
  oracle::Tensor step_e, step_h, step_l;
  CHECK(whole.embed(std::vector<int32_t>{next}, &step_e).ok());
  st = whole.forward(step_e.payload, kv_whole, &step_h);
  if (!st) {
    std::cerr << "decode: " << st.message << "\n";
    return 1;
  }
  CHECK(kv_whole.seq_len == prompt.size() + 1);
  CHECK(whole.lm_head(step_h.payload, &step_l).ok());
  const auto step = logits_of(step_l);
  CHECK(step.size() == m.n_vocab);
  for (float v : step) {
    CHECK(std::isfinite(v));
  }

  // Two prompts must not depend on each other: a reset cache reproduces run 1.
  kv_whole.reset();
  oracle::Tensor r_h, r_l;
  CHECK(whole.forward(embedded.payload, kv_whole, &r_h).ok());
  CHECK(whole.lm_head(r_h.payload, &r_l).ok());
  const auto repeat = logits_of(r_l);
  for (size_t i = 0; i < repeat.size(); ++i) {
    CHECK(std::abs(repeat[i] - single[i]) < 1e-5f);
  }

  std::remove(path.c_str());
  std::cout << "test_gguf ok next_token=" << next << "\n";
  return 0;
}

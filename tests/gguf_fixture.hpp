#pragma once

// Builds a tiny but structurally complete llama-architecture GGUF file on disk
// so the loader, the dequantisers, the tokeniser and the transformer graph can
// all be tested without downloading a multi-gigabyte model.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace oracle_test {

struct TinyModel {
  uint32_t n_layers = 2;
  uint32_t n_embd = 32;
  uint32_t n_ff = 64;
  uint32_t n_heads = 4;
  uint32_t n_kv_heads = 2;
  uint32_t head_dim = 8;
  uint32_t context_length = 128;
  uint32_t n_vocab = 0;  // filled by build()
};

class GgufWriter {
 public:
  void kv_string(const std::string& k, const std::string& v) {
    key(k);
    u32(8);
    str(v);
    ++n_kv_;
  }
  void kv_u32(const std::string& k, uint32_t v) {
    key(k);
    u32(4);
    u32(v);
    ++n_kv_;
  }
  void kv_f32(const std::string& k, float v) {
    key(k);
    u32(6);
    raw(&v, 4);
    ++n_kv_;
  }
  void kv_bool(const std::string& k, bool v) {
    key(k);
    u32(7);
    const uint8_t b = v ? 1 : 0;
    raw(&b, 1);
    ++n_kv_;
  }
  void kv_str_array(const std::string& k, const std::vector<std::string>& vals) {
    key(k);
    u32(9);
    u32(8);
    u64(vals.size());
    for (const auto& s : vals) {
      str(s);
    }
    ++n_kv_;
  }
  void kv_f32_array(const std::string& k, const std::vector<float>& vals) {
    key(k);
    u32(9);
    u32(6);
    u64(vals.size());
    for (float f : vals) {
      raw(&f, 4);
    }
    ++n_kv_;
  }
  void kv_i32_array(const std::string& k, const std::vector<int32_t>& vals) {
    key(k);
    u32(9);
    u32(5);
    u64(vals.size());
    for (int32_t v : vals) {
      raw(&v, 4);
    }
    ++n_kv_;
  }

  // `ne` is in ggml order: ne[0] is the row length.
  void tensor_f32(const std::string& name, const std::vector<uint64_t>& ne,
                  const std::vector<float>& data) {
    Tensor t;
    t.name = name;
    t.ne = ne;
    t.bytes.resize(data.size() * 4);
    std::memcpy(t.bytes.data(), data.data(), data.size() * 4);
    tensors_.push_back(std::move(t));
  }

  bool write(const std::string& path) {
    std::string body;
    body.swap(kv_blob_);
    // Tensor directory, then a 32-byte aligned data section.
    std::string dir;
    uint64_t offset = 0;
    std::vector<uint64_t> offsets;
    for (const auto& t : tensors_) {
      offsets.push_back(offset);
      offset += (t.bytes.size() + 31) & ~static_cast<uint64_t>(31);
    }
    for (size_t i = 0; i < tensors_.size(); ++i) {
      const auto& t = tensors_[i];
      append_str(dir, t.name);
      append_u32(dir, static_cast<uint32_t>(t.ne.size()));
      for (uint64_t d : t.ne) {
        append_u64(dir, d);
      }
      append_u32(dir, 0);  // GGML_F32
      append_u64(dir, offsets[i]);
    }

    std::string head;
    append_u32(head, 0x46554747u);  // "GGUF"
    append_u32(head, 3);
    append_u64(head, tensors_.size());
    append_u64(head, n_kv_);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
      return false;
    }
    out.write(head.data(), static_cast<std::streamsize>(head.size()));
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    out.write(dir.data(), static_cast<std::streamsize>(dir.size()));
    const uint64_t pos = head.size() + body.size() + dir.size();
    const uint64_t pad = ((pos + 31) & ~static_cast<uint64_t>(31)) - pos;
    const std::string zeros(static_cast<size_t>(pad), '\0');
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    for (const auto& t : tensors_) {
      out.write(reinterpret_cast<const char*>(t.bytes.data()),
                static_cast<std::streamsize>(t.bytes.size()));
      const size_t tpad = ((t.bytes.size() + 31) & ~static_cast<size_t>(31)) - t.bytes.size();
      const std::string tz(tpad, '\0');
      out.write(tz.data(), static_cast<std::streamsize>(tz.size()));
    }
    return static_cast<bool>(out);
  }

 private:
  struct Tensor {
    std::string name;
    std::vector<uint64_t> ne;
    std::vector<uint8_t> bytes;
  };

  static void append_u32(std::string& s, uint32_t v) { s.append(reinterpret_cast<char*>(&v), 4); }
  static void append_u64(std::string& s, uint64_t v) { s.append(reinterpret_cast<char*>(&v), 8); }
  static void append_str(std::string& s, const std::string& v) {
    append_u64(s, v.size());
    s += v;
  }

  void key(const std::string& k) { append_str(kv_blob_, k); }
  void u32(uint32_t v) { append_u32(kv_blob_, v); }
  void u64(uint64_t v) { append_u64(kv_blob_, v); }
  void str(const std::string& v) { append_str(kv_blob_, v); }
  void raw(const void* p, size_t n) { kv_blob_.append(static_cast<const char*>(p), n); }

  std::string kv_blob_;
  uint64_t n_kv_ = 0;
  std::vector<Tensor> tensors_;
};

// Deterministic small pseudo-random weights; no <random> so the fixture is
// byte-identical on every platform.
inline std::vector<float> weights(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed * 2654435761u + 1;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * 0.25f;
  }
  return v;
}

inline std::vector<std::string> tiny_vocab() {
  std::vector<std::string> v{"<unk>", "<s>", "</s>"};
  char buf[16];
  for (int i = 0; i < 256; ++i) {
    std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
    v.emplace_back(buf);
  }
  // U+2581 is SentencePiece's word-boundary marker; split from the following
  // letter so the compiler does not fold it into one over-long hex escape.
  // A real SPM vocabulary contains every intermediate prefix, because merging
  // proceeds one bigram at a time -- the fixture has to as well or nothing
  // above single bytes can ever merge.
#define SP "\xE2\x96\x81"
  const char* words[] = {SP "h",    SP "he",   SP "hel",  SP "hell", SP "hello",
                         SP "w",    SP "wo",   SP "wor",  SP "worl", SP "world",
                         SP "t",    SP "th",   SP "the",  SP "a",    "!",
                         SP};
#undef SP
  for (const char* w : words) {
    v.emplace_back(w);
  }
  return v;
}

// Writes the fixture to `path`.  Returns the model geometry actually used.
inline TinyModel build_tiny_gguf(const std::string& path) {
  TinyModel m;
  const auto vocab = tiny_vocab();
  m.n_vocab = static_cast<uint32_t>(vocab.size());

  GgufWriter w;
  w.kv_string("general.architecture", "llama");
  w.kv_string("general.name", "oracle-tiny-test");
  w.kv_u32("general.file_type", 0);
  w.kv_u32("llama.block_count", m.n_layers);
  w.kv_u32("llama.embedding_length", m.n_embd);
  w.kv_u32("llama.feed_forward_length", m.n_ff);
  w.kv_u32("llama.attention.head_count", m.n_heads);
  w.kv_u32("llama.attention.head_count_kv", m.n_kv_heads);
  w.kv_u32("llama.context_length", m.context_length);
  w.kv_u32("llama.rope.dimension_count", m.head_dim);
  w.kv_f32("llama.rope.freq_base", 10000.0f);
  w.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);

  w.kv_string("tokenizer.ggml.model", "llama");
  w.kv_str_array("tokenizer.ggml.tokens", vocab);
  std::vector<float> scores(vocab.size(), 0.f);
  std::vector<int32_t> types(vocab.size(), 1);
  types[0] = 2;  // UNKNOWN
  types[1] = 3;  // CONTROL
  types[2] = 3;  // CONTROL
  for (size_t i = 3; i < 3 + 256; ++i) {
    types[i] = 6;        // BYTE
    scores[i] = -100.f;  // byte fallback must lose to any real merge
  }
  for (size_t i = 3 + 256; i < vocab.size(); ++i) {
    scores[i] = 1.0f + static_cast<float>(i);
  }
  w.kv_f32_array("tokenizer.ggml.scores", scores);
  w.kv_i32_array("tokenizer.ggml.token_type", types);
  w.kv_u32("tokenizer.ggml.bos_token_id", 1);
  w.kv_u32("tokenizer.ggml.eos_token_id", 2);
  w.kv_u32("tokenizer.ggml.unknown_token_id", 0);
  w.kv_bool("tokenizer.ggml.add_bos_token", true);

  const uint32_t q_dim = m.n_heads * m.head_dim;
  const uint32_t kv_dim = m.n_kv_heads * m.head_dim;
  w.tensor_f32("token_embd.weight", {m.n_embd, m.n_vocab},
               weights(static_cast<size_t>(m.n_embd) * m.n_vocab, 1));
  for (uint32_t l = 0; l < m.n_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    w.tensor_f32(p + "attn_norm.weight", {m.n_embd}, std::vector<float>(m.n_embd, 1.0f));
    w.tensor_f32(p + "attn_q.weight", {m.n_embd, q_dim}, weights(m.n_embd * q_dim, 10 + l));
    w.tensor_f32(p + "attn_k.weight", {m.n_embd, kv_dim}, weights(m.n_embd * kv_dim, 20 + l));
    w.tensor_f32(p + "attn_v.weight", {m.n_embd, kv_dim}, weights(m.n_embd * kv_dim, 30 + l));
    w.tensor_f32(p + "attn_output.weight", {q_dim, m.n_embd}, weights(q_dim * m.n_embd, 40 + l));
    w.tensor_f32(p + "ffn_norm.weight", {m.n_embd}, std::vector<float>(m.n_embd, 1.0f));
    w.tensor_f32(p + "ffn_gate.weight", {m.n_embd, m.n_ff}, weights(m.n_embd * m.n_ff, 50 + l));
    w.tensor_f32(p + "ffn_up.weight", {m.n_embd, m.n_ff}, weights(m.n_embd * m.n_ff, 60 + l));
    w.tensor_f32(p + "ffn_down.weight", {m.n_ff, m.n_embd}, weights(m.n_ff * m.n_embd, 70 + l));
  }
  w.tensor_f32("output_norm.weight", {m.n_embd}, std::vector<float>(m.n_embd, 1.0f));
  w.tensor_f32("output.weight", {m.n_embd, m.n_vocab},
               weights(static_cast<size_t>(m.n_embd) * m.n_vocab, 99));

  if (!w.write(path)) {
    m.n_vocab = 0;
  }
  return m;
}

}  // namespace oracle_test

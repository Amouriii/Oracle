#pragma once

// Executes a shard of a llama-family GGUF model.
//
// Weights stay mapped in their quantised form; a row is expanded to f32 inside
// the mat-vec inner loop and thrown away, so resident memory tracks the file
// size rather than the f32 expansion.  A node configured with layers [a, b)
// touches only `blk.a` .. `blk.b-1` (plus the embedding on the first node and
// the output head on the last), which is what makes a model larger than any one
// machine's RAM run across the mesh.

#include "oracle/model/gguf.hpp"
#include "oracle/model/tokenizer.hpp"
#include "oracle/runner/node_runner.hpp"

#include <memory>
#include <string>
#include <vector>

namespace oracle {

// A weight matrix as [rows=out_features, cols=in_features] inside the mapping.
struct WeightView {
  const uint8_t* data{nullptr};
  uint32_t type{0};
  int64_t rows{0};
  int64_t cols{0};
  int64_t row_bytes{0};

  [[nodiscard]] bool valid() const noexcept { return data != nullptr && rows > 0 && cols > 0; }
};

// y(rows) = W * x(cols).  Dequantises one row at a time across the compute pool.
Status weight_matvec(const WeightView& w, const float* x, float* y);

struct GgufLayer {
  std::vector<float> attn_norm;
  std::vector<float> ffn_norm;
  WeightView wq, wk, wv, wo;
  WeightView w_gate, w_up, w_down;
  std::vector<float> bq, bk, bv, bo;
  bool loaded{false};
};

// Per-call working buffers.  Kept out of the runner object so several requests
// can be in flight through the same weights at once: the mapped weights are
// read-only, and nothing else is shared.
struct GgufScratch {
  std::vector<float> x, xb, xb2, q, k, v, attn_out, hb, hb2;
  void resize(uint32_t n_embd, uint32_t n_ff, uint32_t q_dim, uint32_t kv_dim);
};

class GgufRunner final : public NodeRunner {
 public:
  GgufRunner();
  ~GgufRunner() override;

  Status load_layers(const ModelMeta& model, LayerRange layers, std::string_view weights_path) override;
  Status load_layer_blob(uint32_t layer, std::span<const std::byte> blob) override;
  // `in` is [n_tokens, hidden] in the model's activation dtype; `out` is the
  // same shape after this node's layers.  Positions continue from kv.seq_len.
  Status forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) override;
  Status embed(std::span<const int32_t> tokens, Tensor* out) override;
  // Consumes [n_tokens, hidden] and returns f32 logits for the final token.
  Status lm_head(std::span<const std::byte> hidden, Tensor* logits) override;

  [[nodiscard]] const char* name() const noexcept override { return "gguf"; }
  [[nodiscard]] LayerRange layers() const noexcept override { return layers_; }

  [[nodiscard]] const model::ModelInfo& info() const noexcept { return info_; }
  [[nodiscard]] const model::Tokenizer& tokenizer() const noexcept { return tokenizer_; }
  [[nodiscard]] bool owns_embedding() const noexcept { return embed_.valid(); }
  [[nodiscard]] bool owns_output_head() const noexcept { return output_.valid(); }
  [[nodiscard]] uint64_t resident_weight_bytes() const noexcept { return resident_bytes_; }

  // Bytes of KV cache this node needs for `max_seq` tokens of one sequence.
  [[nodiscard]] uint64_t kv_bytes(uint32_t max_seq, DType dtype) const;

 private:
  Status bind_weights();
  Status forward_token(const float* in, float* out, KvCache& kv, uint32_t pos, GgufScratch& s) const;

  model::GgufFile file_;
  model::ModelInfo info_{};
  model::Tokenizer tokenizer_{};
  ModelMeta meta_{};
  LayerRange layers_{};
  std::vector<GgufLayer> blocks_;
  WeightView embed_{};
  WeightView output_{};
  std::vector<float> output_norm_;
  uint64_t resident_bytes_{0};
};

std::unique_ptr<NodeRunner> make_gguf_runner();

}  // namespace oracle

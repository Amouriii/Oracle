#pragma once

#include "oracle/shard/memory_shard_manager.hpp"
#include "oracle/types.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oracle {

struct KvCache {
  KvLayout layout{};
  std::vector<std::byte> k;
  std::vector<std::byte> v;
  uint32_t seq_len{0};

  Status allocate(const KvLayout& l);
  void reset();
};

class NodeRunner {
 public:
  virtual ~NodeRunner() = default;

  virtual Status load_layers(const ModelMeta& model, LayerRange layers,
                             std::string_view weights_path) = 0;
  virtual Status load_layer_blob(uint32_t layer, std::span<const std::byte> blob) = 0;
  virtual Status forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) = 0;
  virtual Status embed(std::span<const int32_t> tokens, Tensor* out) = 0;
  virtual Status lm_head(std::span<const std::byte> hidden, Tensor* logits) = 0;
  [[nodiscard]] virtual const char* name() const noexcept = 0;
  [[nodiscard]] virtual LayerRange layers() const noexcept = 0;
};

class AccelerateRunner final : public NodeRunner {
 public:
  Status load_layers(const ModelMeta& model, LayerRange layers,
                     std::string_view weights_path) override;
  Status load_layer_blob(uint32_t layer, std::span<const std::byte> blob) override;
  Status forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) override;
  Status embed(std::span<const int32_t> tokens, Tensor* out) override;
  Status lm_head(std::span<const std::byte> hidden, Tensor* logits) override;
  [[nodiscard]] const char* name() const noexcept override { return "accelerate"; }
  [[nodiscard]] LayerRange layers() const noexcept override { return layers_; }

  Status gemm_f32(uint32_t m, uint32_t n, uint32_t k, const float* a, const float* b, float* c,
                  float alpha = 1.f, float beta = 0.f);

 private:
  ModelMeta model_{};
  LayerRange layers_{};
  std::vector<std::vector<float>> w_in_;
  std::vector<std::vector<float>> w_out_;
  std::vector<float> embed_w_;
  std::vector<float> lm_w_;
};

class MetalNodeRunner final : public NodeRunner {
 public:
  MetalNodeRunner();
  ~MetalNodeRunner() override;
  Status load_layers(const ModelMeta& model, LayerRange layers,
                     std::string_view weights_path) override;
  Status load_layer_blob(uint32_t layer, std::span<const std::byte> blob) override;
  Status forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) override;
  Status embed(std::span<const int32_t> tokens, Tensor* out) override;
  Status lm_head(std::span<const std::byte> hidden, Tensor* logits) override;
  [[nodiscard]] const char* name() const noexcept override { return "metal"; }
  [[nodiscard]] LayerRange layers() const noexcept override { return layers_; }

  Status gemm_f32(uint32_t m, uint32_t n, uint32_t k, const float* a, const float* b, float* c);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ModelMeta model_{};
  LayerRange layers_{};
};

class LlamaCppRunner final : public NodeRunner {
 public:
  LlamaCppRunner();
  ~LlamaCppRunner() override;
  Status load_layers(const ModelMeta& model, LayerRange layers,
                     std::string_view weights_path) override;
  Status load_layer_blob(uint32_t layer, std::span<const std::byte> blob) override;
  Status forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) override;
  Status embed(std::span<const int32_t> tokens, Tensor* out) override;
  Status lm_head(std::span<const std::byte> hidden, Tensor* logits) override;
  [[nodiscard]] const char* name() const noexcept override { return "llamacpp"; }
  [[nodiscard]] LayerRange layers() const noexcept override { return layers_; }

  static Status read_gguf_meta(const std::string& path, ModelMeta* out);
  Status spawn_rpc(const std::string& binary, uint16_t port);

 private:
  ModelMeta model_{};
  LayerRange layers_{};
  std::string weights_path_;
  int rpc_pid_{-1};
  int rpc_port_{0};
  bool library_linked_{false};
};

std::unique_ptr<NodeRunner> make_accelerate_runner();
std::unique_ptr<NodeRunner> make_metal_runner();
std::unique_ptr<NodeRunner> make_llamacpp_runner();
std::unique_ptr<NodeRunner> make_runner(std::string_view kind);

}  // namespace oracle

// Non-Apple build of MetalNodeRunner.  Metal only exists on Apple platforms, so
// on Linux the runner reports not_implemented and callers fall back to the CPU
// path.  Keeping the type present (rather than #ifdef-ing it out of the header)
// means the orchestrator, tests and `--runner metal` all stay source-compatible.
#include "oracle/runner/node_runner.hpp"

namespace oracle {

struct MetalNodeRunner::Impl {};

MetalNodeRunner::MetalNodeRunner() = default;
MetalNodeRunner::~MetalNodeRunner() = default;

namespace {
Status unsupported() {
  return Status::fail(Errc::not_implemented, "Metal is only available on Apple platforms");
}
}  // namespace

Status MetalNodeRunner::load_layers(const ModelMeta& model, LayerRange layers, std::string_view) {
  model_ = model;
  layers_ = layers;
  return unsupported();
}

Status MetalNodeRunner::load_layer_blob(uint32_t, std::span<const std::byte>) { return unsupported(); }
Status MetalNodeRunner::forward(std::span<const std::byte>, KvCache&, Tensor*) { return unsupported(); }
Status MetalNodeRunner::embed(std::span<const int32_t>, Tensor*) { return unsupported(); }
Status MetalNodeRunner::lm_head(std::span<const std::byte>, Tensor*) { return unsupported(); }
Status MetalNodeRunner::gemm_f32(uint32_t, uint32_t, uint32_t, const float*, const float*, float*) {
  return unsupported();
}

std::unique_ptr<NodeRunner> make_metal_runner() { return std::make_unique<MetalNodeRunner>(); }

}  // namespace oracle

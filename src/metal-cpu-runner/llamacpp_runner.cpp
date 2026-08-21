// llama.cpp integration.
//
// Oracle reads llama.cpp's own container (GGUF) and implements its block
// quantisation formats directly, so `--runner llamacpp` executes the model
// in-process through GgufRunner rather than shelling out.  spawn_rpc() remains
// available for deployments that want to front a real llama.cpp server process
// on a node instead.
#include "oracle/runner/gguf_runner.hpp"
#include "oracle/runner/node_runner.hpp"

#include <cstring>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace oracle {

LlamaCppRunner::LlamaCppRunner() = default;

LlamaCppRunner::~LlamaCppRunner() {
  if (rpc_pid_ > 0) {
    kill(rpc_pid_, SIGTERM);
    int status = 0;
    waitpid(rpc_pid_, &status, 0);
    rpc_pid_ = -1;
  }
}

Status LlamaCppRunner::read_gguf_meta(const std::string& path, ModelMeta* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null meta");
  }
  model::ModelInfo info;
  auto st = model::inspect_gguf(path, &info);
  if (!st) {
    return st;
  }
  model::apply_to_model_meta(info, out);
  return Status::OK();
}

Status LlamaCppRunner::spawn_rpc(const std::string& binary, uint16_t port) {
  if (binary.empty()) {
    return Status::fail(Errc::not_found, "llama binary empty");
  }
  const std::string port_s = std::to_string(port);
  pid_t pid = 0;
  char* argv[] = {const_cast<char*>(binary.c_str()), const_cast<char*>("--host"),
                  const_cast<char*>("127.0.0.1"), const_cast<char*>("--port"),
                  const_cast<char*>(port_s.c_str()), nullptr};
  const int rc = posix_spawn(&pid, binary.c_str(), nullptr, nullptr, argv, environ);
  if (rc != 0) {
    return Status::fail(Errc::io, "posix_spawn llama failed");
  }
  rpc_pid_ = pid;
  rpc_port_ = port;
  return Status::OK();
}

Status LlamaCppRunner::load_layers(const ModelMeta& model, LayerRange layers,
                                   std::string_view weights_path) {
  model_ = model;
  layers_ = layers;
  weights_path_ = weights_path.empty() ? model.path : std::string(weights_path);

  if (!weights_path_.empty()) {
    auto gguf = make_gguf_runner();
    auto st = gguf->load_layers(model, layers, weights_path_);
    if (st) {
      impl_ = std::move(gguf);
      if (auto* g = dynamic_cast<GgufRunner*>(impl_.get())) {
        model::apply_to_model_meta(g->info(), &model_);
      }
#if defined(ORACLE_HAS_LLAMA_CPP)
      library_linked_ = true;
#endif
      return Status::OK();
    }
    // A real path that does not load is an error the operator needs to see,
    // rather than something to paper over with the identity runner.
    return st;
  }

  // No weights configured: fall back to the identity runner so the transport
  // and orchestration layers can still be exercised end to end.
  auto acc = make_accelerate_runner();
  auto st = acc->load_layers(model, layers, {});
  if (!st) {
    return st;
  }
  impl_ = std::move(acc);
  return Status::OK();
}

Status LlamaCppRunner::load_layer_blob(uint32_t layer, std::span<const std::byte> blob) {
  return impl_ ? impl_->load_layer_blob(layer, blob) : Status::OK();
}

Status LlamaCppRunner::forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) {
  if (!impl_) {
    return Status::fail(Errc::invalid_argument, "llamacpp runner: load_layers was not called");
  }
  return impl_->forward(in, kv, out);
}

Status LlamaCppRunner::embed(std::span<const int32_t> tokens, Tensor* out) {
  if (!impl_) {
    return Status::fail(Errc::invalid_argument, "llamacpp runner: load_layers was not called");
  }
  return impl_->embed(tokens, out);
}

Status LlamaCppRunner::lm_head(std::span<const std::byte> hidden, Tensor* logits) {
  if (!impl_) {
    return Status::fail(Errc::invalid_argument, "llamacpp runner: load_layers was not called");
  }
  return impl_->lm_head(hidden, logits);
}

std::unique_ptr<NodeRunner> make_llamacpp_runner() { return std::make_unique<LlamaCppRunner>(); }

std::unique_ptr<NodeRunner> make_runner(std::string_view kind) {
  if (kind == "metal") {
    return make_metal_runner();
  }
  if (kind == "gguf") {
    return make_gguf_runner();
  }
  if (kind == "llamacpp" || kind == "llama" || kind == "auto") {
    return make_llamacpp_runner();
  }
  return make_accelerate_runner();
}

}  // namespace oracle

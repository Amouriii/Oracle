#include "oracle/runner/node_runner.hpp"

#include <cstring>
#include <fstream>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace oracle {
namespace {

bool read_u32(std::istream& in, uint32_t* v) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(v), 4));
}
bool read_u64(std::istream& in, uint64_t* v) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(v), 8));
}

bool read_string(std::istream& in, std::string* s) {
  uint64_t n = 0;
  if (!read_u64(in, &n) || n > (1ull << 20)) {
    return false;
  }
  s->assign(static_cast<size_t>(n), '\0');
  return static_cast<bool>(in.read(s->data(), static_cast<std::streamsize>(n)));
}

bool skip_value(std::istream& in, uint32_t t);

bool skip_array(std::istream& in) {
  uint32_t t = 0;
  uint64_t n = 0;
  if (!read_u32(in, &t) || !read_u64(in, &n) || n > (1ull << 24)) {
    return false;
  }
  for (uint64_t i = 0; i < n; ++i) {
    if (!skip_value(in, t)) {
      return false;
    }
  }
  return true;
}

bool skip_value(std::istream& in, uint32_t t) {
  switch (t) {
    case 0:
    case 1:
    case 7:
      return static_cast<bool>(in.ignore(1));
    case 2:
    case 3:
      return static_cast<bool>(in.ignore(2));
    case 4:
    case 5:
    case 6:
      return static_cast<bool>(in.ignore(4));
    case 10:
    case 11:
    case 12:
      return static_cast<bool>(in.ignore(8));
    case 8: {
      std::string tmp;
      return read_string(in, &tmp);
    }
    case 9:
      return skip_array(in);
    default:
      return false;
  }
}

bool read_u64_value(std::istream& in, uint32_t t, uint64_t* out) {
  switch (t) {
    case 0: {
      uint8_t v = 0;
      if (!in.read(reinterpret_cast<char*>(&v), 1)) {
        return false;
      }
      *out = v;
      return true;
    }
    case 4: {
      uint32_t v = 0;
      if (!read_u32(in, &v)) {
        return false;
      }
      *out = v;
      return true;
    }
    case 5: {
      int32_t v = 0;
      if (!in.read(reinterpret_cast<char*>(&v), 4)) {
        return false;
      }
      *out = static_cast<uint64_t>(v);
      return true;
    }
    case 10:
      return read_u64(in, out);
    case 11: {
      int64_t v = 0;
      if (!in.read(reinterpret_cast<char*>(&v), 8)) {
        return false;
      }
      *out = static_cast<uint64_t>(v);
      return true;
    }
    default:
      return skip_value(in, t);
  }
}

}  // namespace

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
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::fail(Errc::not_found, "gguf not found: " + path);
  }
  char magic[4]{};
  in.read(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) {
    return Status::fail(Errc::protocol, "not a GGUF file");
  }
  uint32_t version = 0;
  uint64_t n_tensors = 0, n_kv = 0;
  if (!read_u32(in, &version) || !read_u64(in, &n_tensors) || !read_u64(in, &n_kv)) {
    return Status::fail(Errc::protocol, "truncated GGUF header");
  }
  ModelMeta m;
  m.path = path;
  m.name = path;
  std::string arch = "llama";
  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key;
    uint32_t t = 0;
    if (!read_string(in, &key) || !read_u32(in, &t)) {
      return Status::fail(Errc::protocol, "bad GGUF kv");
    }
    auto setu = [&](uint32_t* dst) {
      uint64_t v = 0;
      if (read_u64_value(in, t, &v)) {
        *dst = static_cast<uint32_t>(v);
        return true;
      }
      return false;
    };
    if (key == "general.architecture" && t == 8) {
      read_string(in, &arch);
      m.name = arch;
    } else if (key.find(".block_count") != std::string::npos) {
      setu(&m.n_layers);
    } else if (key.find(".embedding_length") != std::string::npos) {
      setu(&m.hidden_dim);
    } else if (key.find(".attention.head_count_kv") != std::string::npos) {
      setu(&m.n_kv_heads);
    } else if (key.find(".attention.head_count") != std::string::npos &&
               key.find("head_count_kv") == std::string::npos) {
      setu(&m.n_heads);
    } else if (key.find(".vocab_size") != std::string::npos) {
      setu(&m.n_vocab);
    } else if (key.find(".context_length") != std::string::npos) {
      setu(&m.max_seq);
    } else if (key == "general.parameter_count") {
      uint64_t v = 0;
      if (read_u64_value(in, t, &v)) {
        m.total_weight_bytes = v / 2;
      }
    } else {
      skip_value(in, t);
    }
  }
  if (m.n_heads && m.hidden_dim) {
    m.head_dim = m.hidden_dim / m.n_heads;
  }
  in.seekg(0, std::ios::end);
  const auto fsz = static_cast<uint64_t>(in.tellg());
  if (!m.total_weight_bytes) {
    m.total_weight_bytes = fsz;
  }
  if (m.n_layers) {
    m.bytes_per_layer = m.total_weight_bytes / m.n_layers;
  }
  (void)n_tensors;
  (void)version;
  *out = m;
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

Status LlamaCppRunner::load_layers(const ModelMeta& model, LayerRange layers, std::string_view weights_path) {
  model_ = model;
  layers_ = layers;
  weights_path_ = std::string(weights_path);
  if (!weights_path_.empty()) {
    ModelMeta gg;
    auto st = read_gguf_meta(weights_path_, &gg);
    if (st) {
      if (gg.n_layers) {
        model_.n_layers = gg.n_layers;
      }
      if (gg.hidden_dim) {
        model_.hidden_dim = gg.hidden_dim;
      }
      if (gg.n_vocab) {
        model_.n_vocab = gg.n_vocab;
      }
      if (gg.n_heads) {
        model_.n_heads = gg.n_heads;
      }
      if (gg.n_kv_heads) {
        model_.n_kv_heads = gg.n_kv_heads;
      }
      model_.path = gg.path;
      model_.bytes_per_layer = gg.bytes_per_layer;
      model_.total_weight_bytes = gg.total_weight_bytes;
    }
  }
#if defined(ORACLE_HAS_LLAMA_CPP)
  library_linked_ = true;
#endif
  return Status::OK();
}

Status LlamaCppRunner::load_layer_blob(uint32_t, std::span<const std::byte>) { return Status::OK(); }

Status LlamaCppRunner::forward(std::span<const std::byte> in, KvCache& kv, Tensor* out) {
  // v1: mesh-owned activations. If llama.cpp is not linked, use Accelerate identity layers
  // so pipeline tests still exercise transport. Subprocess RPC is optional via spawn_rpc().
  AccelerateRunner acc;
  auto st = acc.load_layers(model_, layers_, {});
  if (!st) {
    return st;
  }
  return acc.forward(in, kv, out);
}

Status LlamaCppRunner::embed(std::span<const int32_t> tokens, Tensor* out) {
  AccelerateRunner acc;
  auto st = acc.load_layers(model_, layers_, {});
  if (!st) {
    return st;
  }
  return acc.embed(tokens, out);
}

Status LlamaCppRunner::lm_head(std::span<const std::byte> hidden, Tensor* logits) {
  AccelerateRunner acc;
  auto st = acc.load_layers(model_, layers_, {});
  if (!st) {
    return st;
  }
  return acc.lm_head(hidden, logits);
}

std::unique_ptr<NodeRunner> make_llamacpp_runner() { return std::make_unique<LlamaCppRunner>(); }

std::unique_ptr<NodeRunner> make_runner(std::string_view kind) {
  if (kind == "metal") {
    return make_metal_runner();
  }
  if (kind == "llamacpp" || kind == "llama") {
    return make_llamacpp_runner();
  }
  return make_accelerate_runner();
}

}  // namespace oracle

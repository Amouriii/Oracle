// oracle-model-info: recognise a GGUF file and print what Oracle would need to
// run it, including a suggested layer split across a set of RAM budgets.
#include "oracle/model/gguf.hpp"
#include "oracle/model/tokenizer.hpp"
#include "oracle/runner/gguf_runner.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string human(uint64_t bytes) {
  static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(bytes);
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  std::ostringstream os;
  os << std::fixed << std::setprecision(v < 10 ? 2 : 1) << v << " " << kUnits[u];
  return os.str();
}

void usage() {
  std::cout << "oracle-model-info MODEL.gguf [options]\n"
               "  --json          machine-readable output\n"
               "  --tensors       list every tensor with its shape and quantisation\n"
               "  --split GB,..   propose a layer assignment for the given per-node RAM budgets\n"
               "  --logits IDS    run the model over a comma-separated token id list and print\n"
               "                  the resulting logits, one per line (for cross-checking against\n"
               "                  an independent implementation)\n"
               "  --encode TEXT   tokenise TEXT and print the ids\n";
}

// Runs the prompt through the whole model on this process and prints the
// next-token logits.  Used by tests/reference_llama.py to compare Oracle's
// forward pass against an independent NumPy implementation.
int dump_logits(const std::string& path, const std::vector<int32_t>& tokens) {
  oracle::ModelMeta meta;
  meta.path = path;
  meta.act_dtype = oracle::DType::F32;  // no f16 rounding between the stages
  oracle::GgufRunner runner;
  auto st = runner.load_layers(meta, {0, 0}, path);
  if (!st) {
    std::cerr << "load: " << st.message << "\n";
    return 1;
  }
  const auto& info = runner.info();

  oracle::KvLayout kv_layout;
  kv_layout.n_local_layers = info.n_layers;
  kv_layout.n_kv_heads = info.n_kv_heads;
  kv_layout.max_seq = std::max<uint32_t>(info.context_length, static_cast<uint32_t>(tokens.size()) + 1);
  kv_layout.head_dim = info.head_dim;
  kv_layout.dtype = oracle::DType::F32;
  kv_layout.bytes_k = static_cast<uint64_t>(kv_layout.n_local_layers) * kv_layout.n_kv_heads *
                      kv_layout.head_dim * kv_layout.max_seq * 4;
  kv_layout.bytes_v = kv_layout.bytes_k;
  kv_layout.bytes_total = kv_layout.bytes_k + kv_layout.bytes_v;

  oracle::KvCache kv;
  st = kv.allocate(kv_layout);
  if (!st) {
    std::cerr << "kv: " << st.message << "\n";
    return 1;
  }

  oracle::Tensor embedded, hidden, logits;
  st = runner.embed(tokens, &embedded);
  if (!st) {
    std::cerr << "embed: " << st.message << "\n";
    return 1;
  }
  st = runner.forward(embedded.payload, kv, &hidden);
  if (!st) {
    std::cerr << "forward: " << st.message << "\n";
    return 1;
  }
  st = runner.lm_head(hidden.payload, &logits);
  if (!st) {
    std::cerr << "lm_head: " << st.message << "\n";
    return 1;
  }
  const auto* v = reinterpret_cast<const float*>(logits.payload.data());
  const size_t n = logits.payload.size() / 4;
  std::cout << std::setprecision(9);
  for (size_t i = 0; i < n; ++i) {
    std::cout << v[i] << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  bool json = false;
  bool tensors = false;
  bool want_logits = false;
  std::string encode_text;
  bool want_encode = false;
  std::vector<int32_t> logit_tokens;
  std::vector<double> budgets;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) {
      json = true;
    } else if (std::strcmp(argv[i], "--tensors") == 0) {
      tensors = true;
    } else if (std::strcmp(argv[i], "--split") == 0 && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      std::string item;
      while (std::getline(ss, item, ',')) {
        try {
          budgets.push_back(std::stod(item));
        } catch (...) {
          std::cerr << "bad --split value: " << item << "\n";
          return 2;
        }
      }
    } else if (std::strcmp(argv[i], "--logits") == 0 && i + 1 < argc) {
      want_logits = true;
      std::stringstream ss(argv[++i]);
      std::string item;
      while (std::getline(ss, item, ',')) {
        try {
          logit_tokens.push_back(static_cast<int32_t>(std::stol(item)));
        } catch (...) {
          std::cerr << "bad --logits token id: " << item << "\n";
          return 2;
        }
      }
    } else if (std::strcmp(argv[i], "--encode") == 0 && i + 1 < argc) {
      want_encode = true;
      encode_text = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage();
      return 0;
    } else if (path.empty()) {
      path = argv[i];
    }
  }
  if (path.empty()) {
    usage();
    return 2;
  }

  if (want_logits) {
    if (logit_tokens.empty()) {
      std::cerr << "--logits needs at least one token id\n";
      return 2;
    }
    return dump_logits(path, logit_tokens);
  }

  oracle::model::GgufFile f;
  auto st = f.open(path);
  if (!st) {
    std::cerr << "error: " << st.message << "\n";
    return 1;
  }
  oracle::model::ModelInfo info;
  st = oracle::model::inspect_gguf(f, &info);
  if (!st) {
    std::cerr << "error: " << st.message << "\n";
    return 1;
  }

  if (want_encode) {
    oracle::model::Tokenizer tok;
    auto ts = tok.load(f);
    if (!ts) {
      std::cerr << "tokenizer: " << ts.message << "\n";
      return 1;
    }
    const auto ids = tok.encode(encode_text, tok.add_bos_default());
    for (size_t i = 0; i < ids.size(); ++i) {
      std::cout << (i ? "," : "") << ids[i];
    }
    std::cout << "\n";
    return 0;
  }

  if (json) {
    std::cout << info.to_json() << "\n";
    return 0;
  }

  std::cout << "model         " << info.name << "\n";
  std::cout << "path          " << info.path << "\n";
  std::cout << "architecture  " << info.architecture << "\n";
  std::cout << "quantisation  " << info.quantization << " (" << std::fixed << std::setprecision(2)
            << info.bits_per_weight << " bits/weight)\n";
  std::cout << "parameters    " << info.param_count_human() << " (" << info.param_count << ")\n";
  std::cout << "file          " << human(info.file_size_bytes) << ", " << info.tensor_count
            << " tensors, GGUF v" << f.version() << "\n";
  std::cout << "layers        " << info.n_layers << "\n";
  std::cout << "hidden/ffn    " << info.n_embd << " / " << info.n_ff << "\n";
  std::cout << "heads         " << info.n_heads << " q, " << info.n_kv_heads << " kv, head_dim "
            << info.head_dim << "\n";
  std::cout << "vocab/ctx     " << info.n_vocab << " / " << info.context_length << "\n";
  std::cout << "rope          base " << info.rope_freq_base << ", dim " << info.rope_dim
            << ", scale " << info.rope_freq_scale << "\n";
  std::cout << "tokenizer     " << (info.tokenizer_model.empty() ? "(none)" : info.tokenizer_model);
  oracle::model::Tokenizer tok;
  if (tok.load(f).ok()) {
    std::cout << " -> " << tok.kind_name() << ", bos=" << tok.bos() << " eos=" << tok.eos();
    if (!tok.chat_template().empty()) {
      std::cout << ", chat template present";
    }
  }
  std::cout << "\n";

  std::cout << "\nmemory\n";
  std::cout << "  weights            " << human(info.weight_bytes_total) << "\n";
  std::cout << "  per layer (mean)   " << human(info.bytes_per_layer) << "\n";
  std::cout << "  embeddings + head  " << human(info.non_layer_bytes) << "\n";
  std::cout << "  kv per token       " << human(info.kv_bytes_per_token) << " (f16, all layers)\n";
  std::cout << "  kv at full ctx     " << human(info.kv_bytes_per_token * info.context_length) << "\n";
  std::cout << "  activations        " << human(info.activation_bytes) << "\n";
  std::cout << "  recommended RAM    " << human(info.recommended_ram_bytes) << "\n";

  std::cout << "\nquantisation mix\n";
  for (const auto& [name, bytes] : info.type_bytes) {
    const auto it = info.type_histogram.find(name);
    std::cout << "  " << std::left << std::setw(10) << name << std::right << std::setw(6)
              << (it == info.type_histogram.end() ? 0 : it->second) << " tensors  " << human(bytes)
              << "\n";
  }

  if (!info.supported_for_inference) {
    std::cout << "\nNOT RUNNABLE BY ORACLE: " << info.unsupported_reason << "\n";
  } else {
    std::cout << "\nrunnable by Oracle's gguf runner\n";
  }

  if (tensors) {
    std::cout << "\ntensors\n";
    for (const auto& t : f.tensors()) {
      std::cout << "  " << std::left << std::setw(34) << t.name << std::right << std::setw(8)
                << oracle::model::ggml_type_info(t.type).name << "  [";
      for (size_t i = 0; i < t.ne.size(); ++i) {
        std::cout << (i ? ", " : "") << t.ne[i];
      }
      std::cout << "]  " << human(t.nbytes) << "\n";
    }
  }

  if (!budgets.empty() && info.n_layers) {
    std::cout << "\nproposed split across " << budgets.size() << " nodes\n";
    // Weight each node's share by its RAM budget, then keep layers contiguous so
    // activations only ever travel forward one hop.
    double total_budget = 0;
    for (double b : budgets) {
      total_budget += b;
    }
    uint32_t cursor = 0;
    bool feasible = true;
    for (size_t i = 0; i < budgets.size(); ++i) {
      const uint32_t remaining = info.n_layers - cursor;
      uint32_t take = (i + 1 == budgets.size())
                          ? remaining
                          : static_cast<uint32_t>(info.n_layers * (budgets[i] / total_budget) + 0.5);
      take = std::min(take, remaining);
      if (i + 1 < budgets.size() && take == 0 && remaining > 0) {
        take = 1;
      }
      const uint64_t need = info.bytes_per_layer * take +
                            (i == 0 || i + 1 == budgets.size() ? info.non_layer_bytes / 2 : 0) +
                            info.kv_bytes_per_token * take / std::max(1u, info.n_layers) *
                                info.context_length;
      const uint64_t budget_bytes = static_cast<uint64_t>(budgets[i] * (1ull << 30));
      const bool fits = need <= budget_bytes;
      feasible = feasible && fits;
      std::cout << "  node " << i << "  layers [" << cursor << ", " << cursor + take << ")  needs "
                << human(need) << " of " << human(budget_bytes) << (fits ? "  ok" : "  DOES NOT FIT")
                << "\n";
      cursor += take;
    }
    std::cout << (feasible ? "  plan is feasible\n" : "  plan is NOT feasible with these budgets\n");
  }
  return 0;
}

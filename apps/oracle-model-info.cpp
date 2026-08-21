// oracle-model-info: recognise a GGUF file and print what Oracle would need to
// run it, including a suggested layer split across a set of RAM budgets.
#include "oracle/model/gguf.hpp"
#include "oracle/model/tokenizer.hpp"

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
  std::cout << "oracle-model-info MODEL.gguf [--json] [--tensors] [--split GB,GB,...]\n"
               "  --json      machine-readable output\n"
               "  --tensors   list every tensor with its shape and quantisation\n"
               "  --split     propose a layer assignment for the given per-node RAM budgets\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  bool json = false;
  bool tensors = false;
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

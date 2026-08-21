#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "oracle/cluster_config.hpp"
#include "oracle/model/gguf.hpp"
#include "oracle/shard/memory_shard_manager.hpp"
#include "oracle/types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(oracle_engine, m) {
  m.doc() = "Oracle distributed inference engine bindings";
  py::enum_<oracle::DType>(m, "DType")
      .value("F32", oracle::DType::F32)
      .value("F16", oracle::DType::F16)
      .value("I8", oracle::DType::I8);

  m.def("crc32", [](py::bytes b) {
    std::string s = b;
    auto sp = std::span<const std::byte>(reinterpret_cast<const std::byte*>(s.data()), s.size());
    return oracle::crc32(sp);
  });

  m.def("header_size", [] { return sizeof(oracle::TensorHeader); });

  // Recognise a GGUF without starting a node: useful for capacity planning
  // scripts that decide how to split a model before anything is deployed.
  m.def(
      "inspect_model",
      [](const std::string& path) {
        oracle::model::ModelInfo info;
        auto st = oracle::model::inspect_gguf(path, &info);
        if (!st) {
          throw std::runtime_error(st.message);
        }
        py::dict d;
        d["path"] = info.path;
        d["name"] = info.name;
        d["architecture"] = info.architecture;
        d["quantization"] = info.quantization;
        d["parameters"] = info.param_count;
        d["bits_per_weight"] = info.bits_per_weight;
        d["n_layers"] = info.n_layers;
        d["n_embd"] = info.n_embd;
        d["n_ff"] = info.n_ff;
        d["n_heads"] = info.n_heads;
        d["n_kv_heads"] = info.n_kv_heads;
        d["head_dim"] = info.head_dim;
        d["n_vocab"] = info.n_vocab;
        d["context_length"] = info.context_length;
        d["file_size_bytes"] = info.file_size_bytes;
        d["weight_bytes"] = info.weight_bytes_total;
        d["bytes_per_layer"] = info.bytes_per_layer;
        d["kv_bytes_per_token"] = info.kv_bytes_per_token;
        d["recommended_ram_bytes"] = info.recommended_ram_bytes;
        d["supported"] = info.supported_for_inference;
        d["unsupported_reason"] = info.unsupported_reason;
        py::dict types;
        for (const auto& [name, bytes] : info.type_bytes) {
          types[py::str(name)] = bytes;
        }
        d["type_bytes"] = types;
        return d;
      },
      py::arg("path"), "Parse a .gguf file and report what Oracle needs to run it.");

  m.def("plan_cluster", [](const std::string& path) {
    oracle::ClusterConfig cfg;
    auto st = oracle::load_cluster_toml(path, &cfg);
    if (!st) {
      throw std::runtime_error(st.message);
    }
    oracle::MemoryShardManager mgr(cfg);
    oracle::ShardPlan plan;
    st = mgr.plan(&plan);
    if (!st) {
      throw std::runtime_error(st.message);
    }
    py::list nodes;
    for (const auto& n : plan.nodes) {
      py::dict d;
      d["id"] = n.id;
      d["role"] = n.role;
      d["host"] = n.host;
      d["layer_start"] = n.layers.start;
      d["layer_end"] = n.layers.end;
      nodes.append(d);
    }
    py::dict out;
    out["feasible"] = plan.feasible;
    out["reason"] = plan.reason;
    out["weight_bytes"] = plan.weight_bytes_total;
    out["kv_bytes"] = plan.kv_bytes_total;
    out["nodes"] = nodes;
    return out;
  });
}

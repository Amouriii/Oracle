#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "oracle/cluster_config.hpp"
#include "oracle/shard/memory_shard_manager.hpp"
#include "oracle/types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(oracle_engine, m) {
  m.doc() = "Oracle TB3 LLM engine bindings";
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

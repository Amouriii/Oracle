#include "oracle/cluster_config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace oracle {
namespace {

std::string trim(std::string s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
    ++a;
  }
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
    --b;
  }
  s = s.substr(a, b - a);
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    s = s.substr(1, s.size() - 2);
  }
  return s;
}

DType parse_dtype(const std::string& s) {
  if (s == "f16" || s == "F16") {
    return DType::F16;
  }
  if (s == "i8" || s == "I8" || s == "q8" || s == "q4") {
    return DType::I8;
  }
  if (s == "i32" || s == "I32") {
    return DType::I32;
  }
  return DType::F32;
}

}  // namespace

const NodeConfig* ClusterConfig::find(NodeId id) const {
  for (const auto& n : nodes) {
    if (n.id == id) {
      return &n;
    }
  }
  return nullptr;
}

const NodeConfig* ClusterConfig::master() const {
  for (const auto& n : nodes) {
    if (n.role == "master") {
      return &n;
    }
  }
  return nodes.empty() ? nullptr : &nodes.front();
}

Status load_cluster_toml(const std::string& path, ClusterConfig* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null config");
  }
  std::ifstream in(path);
  if (!in) {
    return Status::fail(Errc::not_found, "cannot open " + path);
  }
  ClusterConfig cfg;
  NodeConfig* node = nullptr;
  std::string section;
  std::string line;
  while (std::getline(in, line)) {
    auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    if (line == "[[nodes]]") {
      cfg.nodes.emplace_back();
      node = &cfg.nodes.back();
      node->http_port = cfg.http_port;
      node->transport_port = cfg.transport_port;
      node->heartbeat_port = cfg.heartbeat_port;
      section = "nodes";
      continue;
    }
    if (line.size() >= 2 && line.front() == '[' && line.back() == ']') {
      section = trim(line.substr(1, line.size() - 2));
      node = nullptr;
      continue;
    }
    auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = trim(line.substr(0, eq));
    const std::string val = trim(line.substr(eq + 1));
    auto setu = [&](uint32_t& d) { d = static_cast<uint32_t>(std::stoul(val)); };
    auto set16 = [&](uint16_t& d) { d = static_cast<uint16_t>(std::stoul(val)); };
    if (section == "cluster" || section.empty()) {
      if (key == "name") {
        cfg.name = val;
      } else if (key == "heartbeat_port") {
        set16(cfg.heartbeat_port);
      } else if (key == "http_port") {
        set16(cfg.http_port);
      } else if (key == "transport_port") {
        set16(cfg.transport_port);
      } else if (key == "mtu") {
        setu(cfg.mtu);
      } else if (key == "heartbeat_misses") {
        setu(cfg.heartbeat_misses);
      } else if (key == "heartbeat_interval_ms") {
        setu(cfg.heartbeat_interval_ms);
      }
    } else if (section == "model") {
      if (key == "name") {
        cfg.model.name = val;
      } else if (key == "path") {
        cfg.model.path = val;
      } else if (key == "n_layers") {
        setu(cfg.model.n_layers);
      } else if (key == "hidden_dim") {
        setu(cfg.model.hidden_dim);
      } else if (key == "n_heads") {
        setu(cfg.model.n_heads);
      } else if (key == "n_kv_heads") {
        setu(cfg.model.n_kv_heads);
      } else if (key == "head_dim") {
        setu(cfg.model.head_dim);
      } else if (key == "n_vocab") {
        setu(cfg.model.n_vocab);
      } else if (key == "max_seq") {
        setu(cfg.model.max_seq);
      } else if (key == "dtype") {
        cfg.model.weight_dtype = parse_dtype(val);
      } else if (key == "act_dtype") {
        cfg.model.act_dtype = parse_dtype(val);
      } else if (key == "bytes_per_layer") {
        cfg.model.bytes_per_layer = std::stoull(val);
      } else if (key == "total_weight_bytes") {
        cfg.model.total_weight_bytes = std::stoull(val);
      }
    } else if (section == "nodes" && node) {
      if (key == "id") {
        node->id = static_cast<NodeId>(std::stoul(val));
      } else if (key == "role") {
        node->role = val;
      } else if (key == "host") {
        node->host = val;
      } else if (key == "http_port") {
        set16(node->http_port);
      } else if (key == "transport_port") {
        set16(node->transport_port);
      } else if (key == "heartbeat_port") {
        set16(node->heartbeat_port);
      } else if (key == "ram_budget_gb") {
        node->ram_budget_gb = std::stod(val);
      } else if (key == "vram_budget_gb") {
        node->vram_budget_gb = std::stod(val);
      } else if (key == "layer_start") {
        node->layers.start = static_cast<uint32_t>(std::stoul(val));
      } else if (key == "layer_end") {
        node->layers.end = static_cast<uint32_t>(std::stoul(val));
      }
    }
  }
  for (auto& n : cfg.nodes) {
    if (n.http_port == 0) {
      n.http_port = cfg.http_port;
    }
    if (n.transport_port == 0) {
      n.transport_port = cfg.transport_port;
    }
    if (n.heartbeat_port == 0) {
      n.heartbeat_port = cfg.heartbeat_port;
    }
  }
  *out = std::move(cfg);
  return Status::OK();
}

Status save_cluster_toml(const std::string& path, const ClusterConfig& cfg) {
  std::ofstream out(path);
  if (!out) {
    return Status::fail(Errc::io, "cannot write " + path);
  }
  out << "[cluster]\n";
  out << "name = \"" << cfg.name << "\"\n";
  out << "heartbeat_port = " << cfg.heartbeat_port << "\n";
  out << "http_port = " << cfg.http_port << "\n";
  out << "transport_port = " << cfg.transport_port << "\n";
  out << "mtu = " << cfg.mtu << "\n\n";
  out << "[model]\n";
  out << "name = \"" << cfg.model.name << "\"\n";
  out << "path = \"" << cfg.model.path << "\"\n";
  out << "n_layers = " << cfg.model.n_layers << "\n";
  out << "hidden_dim = " << cfg.model.hidden_dim << "\n";
  out << "n_heads = " << cfg.model.n_heads << "\n";
  out << "n_kv_heads = " << cfg.model.n_kv_heads << "\n";
  out << "head_dim = " << cfg.model.head_dim << "\n";
  out << "n_vocab = " << cfg.model.n_vocab << "\n";
  out << "max_seq = " << cfg.model.max_seq << "\n\n";
  for (const auto& n : cfg.nodes) {
    out << "[[nodes]]\n";
    out << "id = " << n.id << "\n";
    out << "role = \"" << n.role << "\"\n";
    out << "host = \"" << n.host << "\"\n";
    out << "ram_budget_gb = " << n.ram_budget_gb << "\n";
    out << "vram_budget_gb = " << n.vram_budget_gb << "\n";
    out << "layer_start = " << n.layers.start << "\n";
    out << "layer_end = " << n.layers.end << "\n\n";
  }
  return Status::OK();
}

}  // namespace oracle

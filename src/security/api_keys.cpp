#include "oracle/security/api_keys.hpp"

#include "oracle/security/integrity.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace oracle::security {
namespace {

std::string json_escape(std::string_view s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
    }
    o += c;
  }
  return o;
}

std::string trim(std::string s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) {
    return {};
  }
  const auto b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

}  // namespace

std::string ApiKey::to_json() const {
  std::ostringstream os;
  os << "{\"id\":\"" << json_escape(id) << "\",\"name\":\"" << json_escape(name) << "\",\"admin\":"
     << (admin ? "true" : "false") << ",\"enabled\":" << (enabled ? "true" : "false")
     << ",\"requests_per_minute\":" << requests_per_minute
     << ",\"max_completion_tokens\":" << max_completion_tokens
     << ",\"max_concurrent\":" << max_concurrent << "}";
  return os.str();
}

std::string ApiKeyStore::generate_secret() {
  const std::string hex = random_hex(32);
  return hex.empty() ? std::string{} : ("sk-oracle-" + hex);
}

std::string ApiKeyStore::strip_bearer(std::string_view header) {
  std::string s = trim(std::string(header));
  if (s.size() > 7) {
    std::string prefix = s.substr(0, 7);
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (prefix == "bearer ") {
      s = trim(s.substr(7));
    }
  }
  return s;
}

Status ApiKeyStore::add_plaintext(const std::string& id, const std::string& secret, bool admin,
                                  const std::string& name) {
  if (id.empty() || secret.empty()) {
    return Status::fail(Errc::invalid_argument, "api key id and secret must both be set");
  }
  if (secret.size() < 16) {
    return Status::fail(Errc::invalid_argument,
                        "api key '" + id + "' is shorter than 16 characters; refusing to install it");
  }
  ApiKey k;
  k.id = id;
  k.name = name.empty() ? id : name;
  k.sha256 = sha256_hex(secret);
  k.admin = admin;
  return add_hashed(k);
}

Status ApiKeyStore::add_hashed(const ApiKey& key) {
  if (key.id.empty() || key.sha256.size() != 64) {
    return Status::fail(Errc::invalid_argument, "api key needs an id and a 64-char sha256");
  }
  std::lock_guard<std::mutex> g(mu_);
  for (auto& k : keys_) {
    if (k.id == key.id) {
      k = key;
      return Status::OK();
    }
  }
  keys_.push_back(key);
  return Status::OK();
}

Status ApiKeyStore::load_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return Status::fail(Errc::not_found, "cannot open api key file " + path);
  }
  size_t added = 0;
  size_t lineno = 0;
  std::string line;
  while (std::getline(in, line)) {
    ++lineno;
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    std::istringstream ls(line);
    ApiKey k;
    if (!(ls >> k.id >> k.sha256)) {
      return Status::fail(Errc::invalid_argument,
                          path + ":" + std::to_string(lineno) + ": expected '<id> <sha256hex> ...'");
    }
    if (k.sha256.size() != 64) {
      return Status::fail(Errc::invalid_argument,
                          path + ":" + std::to_string(lineno) + ": second field must be a sha256 hex digest");
    }
    k.name = k.id;
    std::string opt;
    while (ls >> opt) {
      if (opt == "admin") {
        k.admin = true;
      } else if (opt == "disabled") {
        k.enabled = false;
      } else if (opt.rfind("rpm=", 0) == 0) {
        k.requests_per_minute = static_cast<uint32_t>(std::strtoul(opt.c_str() + 4, nullptr, 10));
      } else if (opt.rfind("max_tokens=", 0) == 0) {
        k.max_completion_tokens = static_cast<uint32_t>(std::strtoul(opt.c_str() + 11, nullptr, 10));
      } else if (opt.rfind("max_concurrent=", 0) == 0) {
        k.max_concurrent = static_cast<uint32_t>(std::strtoul(opt.c_str() + 15, nullptr, 10));
      } else if (opt.rfind("name=", 0) == 0) {
        k.name = opt.substr(5);
      }
    }
    auto st = add_hashed(k);
    if (!st) {
      return st;
    }
    ++added;
  }
  if (added == 0) {
    return Status::fail(Errc::not_found, path + " contained no usable api keys");
  }
  return Status::OK();
}

Status ApiKeyStore::load_env(const char* var) {
  const char* raw = std::getenv(var);
  if (!raw || !*raw) {
    return Status::fail(Errc::not_found, std::string(var) + " is not set");
  }
  std::stringstream ss(raw);
  std::string item;
  size_t added = 0;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (item.empty()) {
      continue;
    }
    std::vector<std::string> parts;
    std::stringstream is(item);
    std::string p;
    while (std::getline(is, p, ':')) {
      parts.push_back(p);
    }
    if (parts.size() < 2) {
      return Status::fail(Errc::invalid_argument,
                          std::string(var) + ": entries must look like id:secret[:admin]");
    }
    const bool admin = parts.size() > 2 && parts[2] == "admin";
    auto st = add_plaintext(parts[0], parts[1], admin);
    if (!st) {
      return st;
    }
    ++added;
  }
  return added ? Status::OK() : Status::fail(Errc::not_found, std::string(var) + " had no entries");
}

std::optional<ApiKey> ApiKeyStore::authenticate(std::string_view presented) const {
  if (presented.empty()) {
    return std::nullopt;
  }
  const std::string digest = sha256_hex(presented);
  std::lock_guard<std::mutex> g(mu_);
  std::optional<ApiKey> found;
  // Scan every key even after a hit so the response time does not reveal where
  // in the list a credential lives.
  for (const auto& k : keys_) {
    const bool match = constant_time_equals(digest, k.sha256);
    if (match && k.enabled && !found) {
      found = k;
    }
  }
  return found;
}

std::optional<ApiKey> ApiKeyStore::find_by_id(std::string_view id) const {
  std::lock_guard<std::mutex> g(mu_);
  for (const auto& k : keys_) {
    if (k.id == id) {
      return k;
    }
  }
  return std::nullopt;
}

std::vector<ApiKey> ApiKeyStore::list() const {
  std::lock_guard<std::mutex> g(mu_);
  return keys_;
}

size_t ApiKeyStore::size() const {
  std::lock_guard<std::mutex> g(mu_);
  return keys_.size();
}

void ApiKeyStore::clear() {
  std::lock_guard<std::mutex> g(mu_);
  keys_.clear();
}

}  // namespace oracle::security

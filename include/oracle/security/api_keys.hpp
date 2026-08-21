#pragma once

// API-key store for the OpenAI-compatible endpoint and the worker handshake.
//
// Secrets are never held in memory in the clear: a key is stored as its SHA-256
// digest, so a core dump or a leaked config file does not hand over a usable
// credential.

#include "oracle/types.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::security {

struct ApiKey {
  std::string id;           // public identifier shown in logs and the dashboard
  std::string name;         // human label
  std::string sha256;       // hex digest of the secret
  bool admin{false};        // may read /cluster detail and mutate config
  bool enabled{true};
  uint32_t requests_per_minute{0};  // 0 = inherit the server default
  uint32_t max_completion_tokens{0};
  uint32_t max_concurrent{0};

  [[nodiscard]] std::string to_json() const;  // never includes the secret
};

class ApiKeyStore {
 public:
  void set_require_auth(bool required) { require_auth_ = required; }
  [[nodiscard]] bool require_auth() const noexcept { return require_auth_; }

  Status add_plaintext(const std::string& id, const std::string& secret, bool admin = false,
                       const std::string& name = {});
  Status add_hashed(const ApiKey& key);

  // File format, one key per line, '#' comments allowed:
  //   <id> <sha256hex> [admin] [rpm=N] [max_tokens=N] [max_concurrent=N] [name="..."]
  Status load_file(const std::string& path);
  // ORACLE_API_KEYS="id:secret,id2:secret2:admin" -- convenient for Compose.
  Status load_env(const char* var = "ORACLE_API_KEYS");

  // Accepts the raw secret (already stripped of a "Bearer " prefix).  Returns
  // nullopt when the key is unknown or disabled.  Constant time in the secret.
  // Returns a copy so a concurrent reload cannot invalidate the caller's view.
  [[nodiscard]] std::optional<ApiKey> authenticate(std::string_view presented) const;
  [[nodiscard]] std::optional<ApiKey> find_by_id(std::string_view id) const;

  [[nodiscard]] std::vector<ApiKey> list() const;
  [[nodiscard]] size_t size() const;
  void clear();

  // 32 bytes of urandom as hex.  Empty on failure -- never a weak fallback.
  static std::string generate_secret();
  // Pulls "Bearer xyz" / "xyz" out of an Authorization header value.
  static std::string strip_bearer(std::string_view header);

 private:
  mutable std::mutex mu_;
  std::vector<ApiKey> keys_;
  bool require_auth_{true};
};

}  // namespace oracle::security

#pragma once

// The single admission point for anything entering Oracle from outside:
// HTTP callers, and workers joining the mesh.
//
// One object owns the key store, the limiters, the audit log and the integrity
// manifest, so the HTTP layer never has to reimplement policy and the dashboard
// has one place to read status from.

#include "oracle/security/api_keys.hpp"
#include "oracle/security/audit_log.hpp"
#include "oracle/security/integrity.hpp"
#include "oracle/security/rate_limiter.hpp"
#include "oracle/types.hpp"

#include <cstdint>
#include <string>

namespace oracle::security {

struct SecurityConfig {
  // Authentication
  bool require_api_key{true};
  std::string api_key_file;              // <id> <sha256> [admin] ...
  std::string api_key_env{"ORACLE_API_KEYS"};
  bool allow_anonymous_health{true};     // /health and /v1/models stay open

  // Worker handshake
  std::string cluster_secret;            // shared HMAC secret for workers
  std::string cluster_secret_env{"ORACLE_CLUSTER_SECRET"};
  bool require_worker_auth{true};

  // Request limits
  uint64_t max_request_bytes{1ull << 20};
  uint32_t max_prompt_chars{131072};
  uint32_t max_completion_tokens{2048};
  uint32_t max_messages{256};
  uint32_t max_concurrent_requests{32};
  uint32_t max_concurrent_per_key{8};
  RateLimitConfig rate{};

  // Observability and integrity
  std::string audit_log_path;
  std::string model_manifest_path;
  bool verify_model_integrity{false};
  bool echo_security_log{false};
};

struct RequestIdentity {
  std::string key_id{"anonymous"};
  std::string client;
  bool authenticated{false};
  bool admin{false};
  uint32_t rpm_override{0};
  uint32_t max_tokens_override{0};
  uint32_t max_concurrent_override{0};
};

struct Decision {
  bool allowed{false};
  int http_status{200};
  std::string error_code;  // OpenAI-style: invalid_api_key, rate_limit_exceeded, ...
  std::string message;
  double retry_after_seconds{0};

  static Decision ok() { return Decision{true, 200, {}, {}, 0}; }
};

struct GenerationLimits {
  uint32_t max_tokens{0};
  uint32_t max_prompt_chars{0};
};

class SecurityGate {
 public:
  Status configure(const SecurityConfig& cfg);
  [[nodiscard]] const SecurityConfig& config() const noexcept { return cfg_; }

  ApiKeyStore& keys() noexcept { return keys_; }
  AuditLog& audit() noexcept { return audit_; }
  RateLimiter& limiter() noexcept { return limiter_; }
  ConcurrencyLimiter& concurrency() noexcept { return concurrency_; }
  [[nodiscard]] const IntegrityManifest& manifest() const noexcept { return manifest_; }

  // Step 1: who is calling?  `admin_required` gates the cluster-control routes.
  Decision authenticate(std::string_view authorization_header, const std::string& client_ip,
                        bool admin_required, RequestIdentity* out);
  // Step 2: is the request shaped acceptably and within the caller's budget?
  Decision check_request(const RequestIdentity& id, const std::string& path, uint64_t body_bytes);
  // Step 3: take a concurrency slot for the duration of the generation.
  Decision acquire_slot(const RequestIdentity& id, ConcurrencyLimiter::Lease* lease);

  // Clamps user-supplied generation parameters to what the key is allowed.
  [[nodiscard]] GenerationLimits limits_for(const RequestIdentity& id) const;
  [[nodiscard]] Status validate_generation(const RequestIdentity& id, size_t prompt_chars,
                                           uint32_t requested_tokens, float temperature,
                                           size_t message_count) const;

  // ---- worker handshake --------------------------------------------------
  // Master issues a nonce; the worker returns HMAC(secret, "<node>:<nonce>").
  [[nodiscard]] std::string new_nonce();
  [[nodiscard]] std::string sign(NodeId node, std::string_view nonce) const;
  [[nodiscard]] bool verify(NodeId node, std::string_view nonce, std::string_view signature) const;
  [[nodiscard]] bool worker_auth_configured() const noexcept { return !secret_.empty(); }

  // ---- model integrity ---------------------------------------------------
  // Records the digest when the manifest has no entry yet (trust on first use,
  // logged); fails when a recorded digest does not match.
  Status verify_model_file(const std::string& path);

  [[nodiscard]] std::string status_json() const;

 private:
  SecurityConfig cfg_{};
  ApiKeyStore keys_;
  AuditLog audit_;
  RateLimiter limiter_;
  ConcurrencyLimiter concurrency_;
  IntegrityManifest manifest_;
  std::string secret_;
};

}  // namespace oracle::security

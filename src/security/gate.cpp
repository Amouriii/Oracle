#include "oracle/security/gate.hpp"

#include <cmath>
#include <cstdlib>
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

}  // namespace

Status SecurityGate::configure(const SecurityConfig& cfg) {
  cfg_ = cfg;
  audit_.set_echo_to_stderr(cfg_.echo_security_log);
  if (!cfg_.audit_log_path.empty()) {
    auto st = audit_.open_file(cfg_.audit_log_path);
    if (!st) {
      // A read-only log destination must not take the node down; the ring
      // buffer still backs /cluster and the dashboard.
      audit_.warn("config", "audit_log", st.message);
    }
  }

  keys_.set_require_auth(cfg_.require_api_key);
  // Keys installed programmatically before configure() count too, so an
  // embedder is not forced through a file or an environment variable.
  bool have_keys = keys_.size() > 0;
  if (!cfg_.api_key_file.empty()) {
    auto st = keys_.load_file(cfg_.api_key_file);
    if (!st) {
      return Status::fail(st.code, "api keys: " + st.message);
    }
    have_keys = true;
    audit_.info("config", "api_keys", "loaded " + std::to_string(keys_.size()) + " keys from " +
                                          cfg_.api_key_file);
  }
  if (!cfg_.api_key_env.empty()) {
    auto st = keys_.load_env(cfg_.api_key_env.c_str());
    if (st) {
      have_keys = true;
      audit_.info("config", "api_keys", "loaded keys from $" + cfg_.api_key_env);
    } else if (st.code == Errc::invalid_argument) {
      return Status::fail(st.code, "api keys: " + st.message);
    }
  }
  if (cfg_.require_api_key && !have_keys) {
    return Status::fail(Errc::invalid_argument,
                        "authentication is required but no API keys were configured; set "
                        "security.api_key_file, $" +
                            cfg_.api_key_env + ", or turn off security.require_api_key");
  }

  secret_ = cfg_.cluster_secret;
  if (secret_.empty() && !cfg_.cluster_secret_env.empty()) {
    if (const char* env = std::getenv(cfg_.cluster_secret_env.c_str())) {
      secret_ = env;
    }
  }
  if (cfg_.require_worker_auth && secret_.empty()) {
    return Status::fail(Errc::invalid_argument,
                        "worker authentication is required but no cluster secret was configured; set "
                        "security.cluster_secret or $" +
                            cfg_.cluster_secret_env);
  }

  limiter_.configure(cfg_.rate);
  concurrency_.configure(cfg_.max_concurrent_requests, cfg_.max_concurrent_per_key);

  if (!cfg_.model_manifest_path.empty()) {
    auto st = manifest_.load(cfg_.model_manifest_path);
    if (!st && cfg_.verify_model_integrity) {
      audit_.warn("integrity", "manifest", st.message);
    }
  }
  audit_.info("config", "security",
              std::string("auth=") + (cfg_.require_api_key ? "required" : "open") +
                  " worker_auth=" + (cfg_.require_worker_auth ? "required" : "open") +
                  " rpm=" + std::to_string(cfg_.rate.requests_per_minute) +
                  " max_concurrent=" + std::to_string(cfg_.max_concurrent_requests));
  return Status::OK();
}

Decision SecurityGate::authenticate(std::string_view authorization_header, const std::string& client_ip,
                                    bool admin_required, RequestIdentity* out) {
  RequestIdentity id;
  id.client = client_ip;

  // A banned source is turned away before any hashing work is done.
  if (limiter_.is_banned(client_ip)) {
    Decision d;
    d.http_status = 429;
    d.error_code = "rate_limit_exceeded";
    d.message = "too many rejected requests from this address";
    d.retry_after_seconds = limiter_.retry_after_seconds(client_ip);
    if (out) {
      *out = id;
    }
    return d;
  }

  const std::string secret = ApiKeyStore::strip_bearer(authorization_header);
  if (!secret.empty()) {
    if (const auto key = keys_.authenticate(secret)) {
      id.key_id = key->id;
      id.authenticated = true;
      id.admin = key->admin;
      id.rpm_override = key->requests_per_minute;
      id.max_tokens_override = key->max_completion_tokens;
      id.max_concurrent_override = key->max_concurrent;
    } else {
      limiter_.penalise(client_ip);
      audit_.warn("auth", client_ip, "rejected an unrecognised API key");
      Decision d;
      d.http_status = 401;
      d.error_code = "invalid_api_key";
      d.message = "Incorrect API key provided";
      if (out) {
        *out = id;
      }
      return d;
    }
  }

  if (!id.authenticated && cfg_.require_api_key) {
    limiter_.penalise(client_ip);
    audit_.warn("auth", client_ip, "request without credentials");
    Decision d;
    d.http_status = 401;
    d.error_code = "missing_api_key";
    d.message = "Missing Authorization header; expected 'Authorization: Bearer <key>'";
    if (out) {
      *out = id;
    }
    return d;
  }

  if (admin_required && !id.admin) {
    audit_.warn("auth", id.key_id, "attempted an admin-only route");
    Decision d;
    d.http_status = 403;
    d.error_code = "insufficient_permissions";
    d.message = "This endpoint requires an admin API key";
    if (out) {
      *out = id;
    }
    return d;
  }

  if (out) {
    *out = id;
  }
  return Decision::ok();
}

Decision SecurityGate::check_request(const RequestIdentity& id, const std::string& path,
                                     uint64_t body_bytes) {
  if (cfg_.max_request_bytes && body_bytes > cfg_.max_request_bytes) {
    limiter_.penalise(id.authenticated ? id.key_id : id.client);
    audit_.warn("request", id.key_id,
                "body of " + std::to_string(body_bytes) + " bytes exceeds the " +
                    std::to_string(cfg_.max_request_bytes) + " byte limit on " + path);
    Decision d;
    d.http_status = 413;
    d.error_code = "request_too_large";
    d.message = "Request body exceeds " + std::to_string(cfg_.max_request_bytes) + " bytes";
    return d;
  }

  const std::string subject = id.authenticated ? id.key_id : id.client;
  if (!limiter_.allow(subject, id.rpm_override)) {
    Decision d;
    d.http_status = 429;
    d.error_code = "rate_limit_exceeded";
    d.retry_after_seconds = limiter_.retry_after_seconds(subject);
    d.message = "Rate limit reached for " + subject;
    audit_.warn("rate_limit", subject, "throttled on " + path);
    return d;
  }
  return Decision::ok();
}

Decision SecurityGate::acquire_slot(const RequestIdentity& id, ConcurrencyLimiter::Lease* lease) {
  const std::string subject = id.authenticated ? id.key_id : id.client;
  auto held = concurrency_.acquire(subject, id.max_concurrent_override);
  if (!held.held()) {
    Decision d;
    d.http_status = 503;
    d.error_code = "server_busy";
    d.retry_after_seconds = 1.0;
    d.message = "Oracle is at its concurrency limit (" +
                std::to_string(cfg_.max_concurrent_requests) + " in flight)";
    audit_.warn("capacity", subject, "refused: concurrency limit reached");
    return d;
  }
  if (lease) {
    *lease = std::move(held);
  }
  return Decision::ok();
}

GenerationLimits SecurityGate::limits_for(const RequestIdentity& id) const {
  GenerationLimits l;
  l.max_tokens = id.max_tokens_override ? std::min(id.max_tokens_override, cfg_.max_completion_tokens)
                                        : cfg_.max_completion_tokens;
  l.max_prompt_chars = cfg_.max_prompt_chars;
  return l;
}

Status SecurityGate::validate_generation(const RequestIdentity& id, size_t prompt_chars,
                                         uint32_t requested_tokens, float temperature,
                                         size_t message_count) const {
  const GenerationLimits l = limits_for(id);
  if (l.max_prompt_chars && prompt_chars > l.max_prompt_chars) {
    return Status::fail(Errc::invalid_argument,
                        "prompt of " + std::to_string(prompt_chars) + " characters exceeds the limit of " +
                            std::to_string(l.max_prompt_chars));
  }
  if (cfg_.max_messages && message_count > cfg_.max_messages) {
    return Status::fail(Errc::invalid_argument,
                        "conversation has " + std::to_string(message_count) + " messages; the limit is " +
                            std::to_string(cfg_.max_messages));
  }
  if (l.max_tokens && requested_tokens > l.max_tokens) {
    return Status::fail(Errc::invalid_argument,
                        "max_tokens of " + std::to_string(requested_tokens) + " exceeds the limit of " +
                            std::to_string(l.max_tokens));
  }
  if (!std::isfinite(temperature) || temperature < 0.0f || temperature > 2.0f) {
    return Status::fail(Errc::invalid_argument, "temperature must be between 0 and 2");
  }
  return Status::OK();
}

std::string SecurityGate::new_nonce() {
  std::string n = random_hex(16);
  if (n.empty()) {
    // random_hex only fails when /dev/urandom is unavailable; a predictable
    // nonce would defeat the handshake, so surface it instead of inventing one.
    audit_.alert("worker", "nonce", "/dev/urandom unavailable; cannot issue a worker challenge");
  }
  return n;
}

std::string SecurityGate::sign(NodeId node, std::string_view nonce) const {
  if (secret_.empty()) {
    return {};
  }
  return hmac_sha256_hex(secret_, std::to_string(node) + ":" + std::string(nonce));
}

bool SecurityGate::verify(NodeId node, std::string_view nonce, std::string_view signature) const {
  if (!cfg_.require_worker_auth) {
    return true;
  }
  if (secret_.empty() || nonce.empty() || signature.empty()) {
    return false;
  }
  return constant_time_equals(sign(node, nonce), signature);
}

Status SecurityGate::verify_model_file(const std::string& path) {
  if (path.empty()) {
    return Status::OK();
  }
  const auto* entry = manifest_.find(path);
  if (!entry) {
    if (!cfg_.verify_model_integrity) {
      return Status::OK();
    }
    // First sight of this file: record the digest and say so loudly, so an
    // operator can commit the manifest and get real protection next boot.
    std::string hex;
    uint64_t bytes = 0;
    auto st = sha256_file(path, &hex, &bytes);
    if (!st) {
      return st;
    }
    manifest_.record(IntegrityEntry{path, hex, bytes});
    if (!cfg_.model_manifest_path.empty()) {
      auto save = manifest_.save(cfg_.model_manifest_path);
      if (!save) {
        audit_.warn("integrity", path, save.message);
      }
    }
    audit_.warn("integrity", path, "no recorded digest; trusting on first use and recording " + hex);
    return Status::OK();
  }
  auto st = manifest_.verify(path);
  if (!st) {
    audit_.alert("integrity", path, st.message);
    return st;
  }
  audit_.info("integrity", path, "digest matches the manifest");
  return Status::OK();
}

std::string SecurityGate::status_json() const {
  std::ostringstream os;
  os << "{";
  os << "\"auth_required\":" << (cfg_.require_api_key ? "true" : "false");
  os << ",\"api_keys\":" << keys_.size();
  os << ",\"worker_auth\":" << (cfg_.require_worker_auth ? "true" : "false");
  os << ",\"worker_secret_configured\":" << (secret_.empty() ? "false" : "true");
  os << ",\"integrity_checks\":" << (cfg_.verify_model_integrity ? "true" : "false");
  os << ",\"manifest_entries\":" << manifest_.entries().size();
  os << ",\"limits\":{";
  os << "\"requests_per_minute\":" << cfg_.rate.requests_per_minute;
  os << ",\"burst\":" << cfg_.rate.burst;
  os << ",\"max_request_bytes\":" << cfg_.max_request_bytes;
  os << ",\"max_completion_tokens\":" << cfg_.max_completion_tokens;
  os << ",\"max_concurrent_requests\":" << cfg_.max_concurrent_requests;
  os << ",\"max_concurrent_per_key\":" << cfg_.max_concurrent_per_key;
  os << "}";
  os << ",\"traffic\":{";
  os << "\"allowed\":" << limiter_.total_allowed();
  os << ",\"rejected\":" << limiter_.total_rejected();
  os << ",\"in_flight\":" << concurrency_.in_flight();
  os << ",\"tracked_subjects\":" << limiter_.tracked_subjects();
  os << "}";
  os << ",\"events\":" << audit_.total();
  os << ",\"recent\":[";
  const auto recent = audit_.recent(20);
  for (size_t i = 0; i < recent.size(); ++i) {
    os << (i ? "," : "") << recent[i].to_json();
  }
  os << "]";
  if (!audit_.file_path().empty()) {
    os << ",\"log_file\":\"" << json_escape(audit_.file_path()) << "\"";
  }
  os << "}";
  return os.str();
}

}  // namespace oracle::security

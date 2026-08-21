// Covers the security layer end to end: hashing, key handling, rate limiting,
// concurrency caps, request validation, worker HMAC auth and model integrity.
#include "oracle/security/gate.hpp"

#include "check.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

using namespace oracle;
using namespace oracle::security;

namespace {

void check_keys() {
  ApiKeyStore store;
  CHECK(store.add_plaintext("k1", "sk-oracle-abcdefghijklmnop", true).ok());
  // Short secrets are refused outright rather than silently accepted.
  CHECK(!store.add_plaintext("k2", "short").ok());
  CHECK(store.size() == 1);

  const auto ok = store.authenticate("sk-oracle-abcdefghijklmnop");
  CHECK(ok.has_value());
  CHECK(ok->id == "k1");
  CHECK(ok->admin);
  CHECK(!store.authenticate("sk-oracle-abcdefghijklmnoq").has_value());
  CHECK(!store.authenticate("").has_value());

  // The stored form must never be the secret itself.
  for (const auto& k : store.list()) {
    CHECK(k.sha256.size() == 64);
    CHECK(k.sha256.find("sk-oracle") == std::string::npos);
    CHECK(k.to_json().find("sk-oracle") == std::string::npos);
  }

  CHECK(ApiKeyStore::strip_bearer("Bearer abc") == "abc");
  CHECK(ApiKeyStore::strip_bearer("bearer abc") == "abc");
  CHECK(ApiKeyStore::strip_bearer("  abc ") == "abc");
}

void check_key_file() {
  const auto path = (std::filesystem::temp_directory_path() / "oracle-keys-test").string();
  {
    std::ofstream out(path);
    out << "# comment\n";
    out << "alice " << sha256_hex("sk-oracle-alice-secret-value") << " admin rpm=600\n";
    out << "bob " << sha256_hex("sk-oracle-bob-secret-value") << " max_tokens=64\n";
  }
  ApiKeyStore store;
  auto st = store.load_file(path);
  if (!st) {
    std::cerr << st.message << "\n";
    std::exit(1);
  }
  CHECK(store.size() == 2);
  const auto a = store.authenticate("sk-oracle-alice-secret-value");
  CHECK(a && a->admin && a->requests_per_minute == 600);
  const auto b = store.authenticate("sk-oracle-bob-secret-value");
  CHECK(b && !b->admin && b->max_completion_tokens == 64);

  // A malformed line must be reported, not skipped.
  {
    std::ofstream out(path);
    out << "broken-line-without-a-digest\n";
  }
  ApiKeyStore bad;
  CHECK(!bad.load_file(path).ok());
  std::remove(path.c_str());
}

void check_rate_limit() {
  RateLimitConfig cfg;
  cfg.requests_per_minute = 60;  // 1/s
  cfg.burst = 3;
  cfg.abuse_threshold = 4;
  cfg.ban_window_seconds = 60;
  cfg.ban_seconds = 60;
  RateLimiter rl(cfg);
  rl.configure(cfg);

  CHECK(rl.allow("a"));
  CHECK(rl.allow("a"));
  CHECK(rl.allow("a"));
  CHECK(!rl.allow("a"));  // burst exhausted
  CHECK(rl.total_allowed() == 3);
  CHECK(rl.total_rejected() >= 1);
  CHECK(rl.retry_after_seconds("a") > 0.0);

  // A different subject has its own bucket.
  CHECK(rl.allow("b"));

  // Enough rejections in the window earn a ban.
  for (int i = 0; i < 10; ++i) {
    (void)rl.allow("a");
  }
  CHECK(rl.is_banned("a"));
  CHECK(!rl.is_banned("b"));
  CHECK(rl.retry_after_seconds("a") > 1.0);
}

void check_concurrency() {
  ConcurrencyLimiter cl;
  cl.configure(2, 1);
  auto l1 = cl.acquire("k1");
  CHECK(l1.held());
  auto l2 = cl.acquire("k1");
  CHECK(!l2.held());  // per-subject cap
  auto l3 = cl.acquire("k2");
  CHECK(l3.held());
  auto l4 = cl.acquire("k3");
  CHECK(!l4.held());  // global cap
  CHECK(cl.in_flight() == 2);
  l1.release();
  CHECK(cl.in_flight() == 1);
  {
    auto l5 = cl.acquire("k4");
    CHECK(l5.held());
    CHECK(cl.in_flight() == 2);
  }
  // Leaving scope must return the slot.
  CHECK(cl.in_flight() == 1);
}

void check_gate() {
  SecurityConfig cfg;
  cfg.require_api_key = true;
  cfg.require_worker_auth = true;
  cfg.cluster_secret = "cluster-secret-for-tests";
  cfg.api_key_env.clear();
  cfg.max_request_bytes = 1024;
  cfg.max_completion_tokens = 128;
  cfg.max_concurrent_requests = 2;
  cfg.max_concurrent_per_key = 2;
  cfg.rate.requests_per_minute = 6000;
  cfg.rate.burst = 100;

  SecurityGate gate;
  // Requiring auth with no keys configured must fail loudly at startup.
  CHECK(!gate.configure(cfg).ok());

  const std::string secret = "sk-oracle-test-secret-1234567890";
  CHECK(gate.keys().add_plaintext("tester", secret, false).ok());
  auto st = gate.configure(cfg);
  if (!st) {
    std::cerr << "configure: " << st.message << "\n";
    std::exit(1);
  }
  // configure() must not have discarded the pre-seeded key.
  CHECK(gate.keys().size() == 1);

  RequestIdentity id;
  auto d = gate.authenticate("", "10.0.0.9", false, &id);
  CHECK(!d.allowed && d.http_status == 401 && d.error_code == "missing_api_key");

  d = gate.authenticate("Bearer nope-nope-nope-nope", "10.0.0.9", false, &id);
  CHECK(!d.allowed && d.http_status == 401 && d.error_code == "invalid_api_key");

  d = gate.authenticate("Bearer " + secret, "10.0.0.9", false, &id);
  CHECK(d.allowed && id.authenticated && id.key_id == "tester" && !id.admin);

  // Admin-only routes reject a non-admin key.
  RequestIdentity admin_id;
  d = gate.authenticate("Bearer " + secret, "10.0.0.9", true, &admin_id);
  CHECK(!d.allowed && d.http_status == 403);

  d = gate.check_request(id, "/v1/chat/completions", 64);
  CHECK(d.allowed);
  d = gate.check_request(id, "/v1/chat/completions", 4096);
  CHECK(!d.allowed && d.http_status == 413);

  // Generation parameter validation.
  CHECK(gate.validate_generation(id, 10, 32, 0.7f, 2).ok());
  CHECK(!gate.validate_generation(id, 10, 100000, 0.7f, 2).ok());
  CHECK(!gate.validate_generation(id, 10, 32, -1.0f, 2).ok());
  CHECK(!gate.validate_generation(id, 10, 32, 5.0f, 2).ok());
  CHECK(!gate.validate_generation(id, 1u << 30, 32, 0.7f, 2).ok());

  // Concurrency slots.
  ConcurrencyLimiter::Lease a, b, c;
  CHECK(gate.acquire_slot(id, &a).allowed);
  CHECK(gate.acquire_slot(id, &b).allowed);
  auto busy = gate.acquire_slot(id, &c);
  CHECK(!busy.allowed && busy.http_status == 503);
  a.release();
  CHECK(gate.acquire_slot(id, &c).allowed);

  // Worker handshake.
  const std::string nonce = gate.new_nonce();
  CHECK(nonce.size() == 32);
  const std::string sig = gate.sign(7, nonce);
  CHECK(!sig.empty());
  CHECK(gate.verify(7, nonce, sig));
  CHECK(!gate.verify(8, nonce, sig));           // wrong node
  CHECK(!gate.verify(7, gate.new_nonce(), sig));  // wrong nonce
  CHECK(!gate.verify(7, nonce, "deadbeef"));    // wrong signature
  CHECK(!gate.verify(7, nonce, ""));

  const std::string status = gate.status_json();
  CHECK(status.find("\"auth_required\":true") != std::string::npos);
  CHECK(status.find("\"worker_secret_configured\":true") != std::string::npos);
  CHECK(status.find(secret) == std::string::npos);
  CHECK(gate.audit().total() > 0);
}

void check_integrity() {
  const auto dir = std::filesystem::temp_directory_path();
  const auto model = (dir / "oracle-integrity-test.bin").string();
  const auto manifest = (dir / "oracle-integrity-test.manifest").string();
  {
    std::ofstream out(model, std::ios::binary);
    out << "the quick brown fox";
  }
  std::string hex;
  uint64_t bytes = 0;
  CHECK(sha256_file(model, &hex, &bytes).ok());
  CHECK(bytes == 19);
  CHECK(hex == sha256_hex("the quick brown fox"));

  IntegrityManifest m;
  m.record(IntegrityEntry{model, hex, bytes});
  CHECK(m.save(manifest).ok());
  IntegrityManifest loaded;
  CHECK(loaded.load(manifest).ok());
  CHECK(loaded.verify(model).ok());
  CHECK(loaded.verify("/no/such/file").code == Errc::not_found);

  // Tampering with the file must be caught.
  {
    std::ofstream out(model, std::ios::binary);
    out << "the quick brown cat";
  }
  const auto bad = loaded.verify(model);
  CHECK(!bad.ok() && bad.code == Errc::protocol);

  std::remove(model.c_str());
  std::remove(manifest.c_str());
}

}  // namespace

int main() {
  check_keys();
  check_key_file();
  check_rate_limit();
  check_concurrency();
  check_gate();
  check_integrity();
  std::cout << "test_security ok\n";
  return 0;
}

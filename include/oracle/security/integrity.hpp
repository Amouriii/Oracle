#pragma once

// SHA-256, HMAC-SHA256 and constant-time comparison.
//
// Implemented in-tree rather than pulled from OpenSSL so that a worker image
// stays a single static binary with no TLS dependency; these are the only
// primitives Oracle needs (API-key hashing, worker handshake signatures and
// model-file integrity).

#include "oracle/types.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace oracle::security {

using Digest = std::array<uint8_t, 32>;

class Sha256 {
 public:
  Sha256();
  void update(std::span<const uint8_t> data);
  void update(std::string_view s);
  Digest finish();

 private:
  void block(const uint8_t* p);

  uint32_t h_[8]{};
  uint8_t buf_[64]{};
  size_t buf_len_{0};
  uint64_t total_{0};
};

Digest sha256(std::span<const uint8_t> data);
Digest sha256(std::string_view s);
std::string to_hex(const Digest& d);
std::string sha256_hex(std::string_view s);

// Streams the file in chunks so a 40 GiB model does not have to be resident.
Status sha256_file(const std::string& path, std::string* hex_out, uint64_t* bytes_out = nullptr);

Digest hmac_sha256(std::string_view key, std::string_view message);
std::string hmac_sha256_hex(std::string_view key, std::string_view message);

// Length-independent, value-independent comparison for secrets.
bool constant_time_equals(std::string_view a, std::string_view b);

// Cryptographically random bytes from the OS, rendered as lowercase hex.
std::string random_hex(size_t n_bytes);

// A recorded expectation for one file: "<sha256hex>  <path>".
struct IntegrityEntry {
  std::string path;
  std::string sha256;
  uint64_t bytes{0};
};

class IntegrityManifest {
 public:
  Status load(const std::string& path);
  Status save(const std::string& path) const;
  void record(const IntegrityEntry& e);
  [[nodiscard]] const IntegrityEntry* find(const std::string& path) const;
  // Hashes `path` and compares against the recorded digest.  Returns not_found
  // when the manifest has no entry, so callers can decide whether that is fatal.
  [[nodiscard]] Status verify(const std::string& path) const;
  [[nodiscard]] const std::vector<IntegrityEntry>& entries() const noexcept { return entries_; }

 private:
  std::vector<IntegrityEntry> entries_;
};

}  // namespace oracle::security

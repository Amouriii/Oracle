#include "oracle/security/integrity.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace oracle::security {
namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256() {
  h_[0] = 0x6a09e667u;
  h_[1] = 0xbb67ae85u;
  h_[2] = 0x3c6ef372u;
  h_[3] = 0xa54ff53au;
  h_[4] = 0x510e527fu;
  h_[5] = 0x9b05688cu;
  h_[6] = 0x1f83d9abu;
  h_[7] = 0x5be0cd19u;
}

void Sha256::block(const uint8_t* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(p[i * 4]) << 24) | (static_cast<uint32_t>(p[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(p[i * 4 + 2]) << 8) | static_cast<uint32_t>(p[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
  uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
    const uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = S0 + maj;
    hh = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += hh;
}

void Sha256::update(std::span<const uint8_t> data) {
  total_ += data.size();
  size_t i = 0;
  if (buf_len_) {
    const size_t take = std::min(sizeof(buf_) - buf_len_, data.size());
    std::memcpy(buf_ + buf_len_, data.data(), take);
    buf_len_ += take;
    i += take;
    if (buf_len_ == sizeof(buf_)) {
      block(buf_);
      buf_len_ = 0;
    }
  }
  for (; i + 64 <= data.size(); i += 64) {
    block(data.data() + i);
  }
  if (i < data.size()) {
    std::memcpy(buf_, data.data() + i, data.size() - i);
    buf_len_ = data.size() - i;
  }
}

void Sha256::update(std::string_view s) {
  update(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(s.data()), s.size()));
}

Digest Sha256::finish() {
  const uint64_t bits = total_ * 8;
  uint8_t pad[72]{};
  pad[0] = 0x80;
  const size_t rem = buf_len_ % 64;
  const size_t pad_len = (rem < 56) ? (56 - rem) : (120 - rem);
  update(std::span<const uint8_t>(pad, pad_len));
  uint8_t len[8];
  for (int i = 0; i < 8; ++i) {
    len[i] = static_cast<uint8_t>((bits >> (56 - i * 8)) & 0xFF);
  }
  update(std::span<const uint8_t>(len, 8));

  Digest out{};
  for (int i = 0; i < 8; ++i) {
    out[i * 4 + 0] = static_cast<uint8_t>((h_[i] >> 24) & 0xFF);
    out[i * 4 + 1] = static_cast<uint8_t>((h_[i] >> 16) & 0xFF);
    out[i * 4 + 2] = static_cast<uint8_t>((h_[i] >> 8) & 0xFF);
    out[i * 4 + 3] = static_cast<uint8_t>(h_[i] & 0xFF);
  }
  return out;
}

Digest sha256(std::span<const uint8_t> data) {
  Sha256 h;
  h.update(data);
  return h.finish();
}

Digest sha256(std::string_view s) {
  Sha256 h;
  h.update(s);
  return h.finish();
}

std::string to_hex(const Digest& d) {
  static const char* kHex = "0123456789abcdef";
  std::string out(64, '0');
  for (size_t i = 0; i < d.size(); ++i) {
    out[i * 2] = kHex[d[i] >> 4];
    out[i * 2 + 1] = kHex[d[i] & 0x0F];
  }
  return out;
}

std::string sha256_hex(std::string_view s) { return to_hex(sha256(s)); }

Status sha256_file(const std::string& path, std::string* hex_out, uint64_t* bytes_out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::fail(Errc::not_found, "cannot open " + path + " for hashing");
  }
  Sha256 h;
  std::vector<char> buf(1u << 20);
  uint64_t total = 0;
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const auto n = static_cast<size_t>(in.gcount());
    if (n == 0) {
      break;
    }
    total += n;
    h.update(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(buf.data()), n));
  }
  if (hex_out) {
    *hex_out = to_hex(h.finish());
  }
  if (bytes_out) {
    *bytes_out = total;
  }
  return Status::OK();
}

Digest hmac_sha256(std::string_view key, std::string_view message) {
  uint8_t k[64]{};
  if (key.size() > 64) {
    const Digest kd = sha256(key);
    std::memcpy(k, kd.data(), kd.size());
  } else {
    std::memcpy(k, key.data(), key.size());
  }
  uint8_t ipad[64], opad[64];
  for (int i = 0; i < 64; ++i) {
    ipad[i] = static_cast<uint8_t>(k[i] ^ 0x36);
    opad[i] = static_cast<uint8_t>(k[i] ^ 0x5c);
  }
  Sha256 inner;
  inner.update(std::span<const uint8_t>(ipad, 64));
  inner.update(message);
  const Digest id = inner.finish();

  Sha256 outer;
  outer.update(std::span<const uint8_t>(opad, 64));
  outer.update(std::span<const uint8_t>(id.data(), id.size()));
  return outer.finish();
}

std::string hmac_sha256_hex(std::string_view key, std::string_view message) {
  return to_hex(hmac_sha256(key, message));
}

bool constant_time_equals(std::string_view a, std::string_view b) {
  // Fold both operands into a fixed-width digest first: that way neither the
  // running time nor the branch pattern depends on the inputs' lengths.
  const Digest da = sha256(a);
  const Digest db = sha256(b);
  uint8_t diff = 0;
  for (size_t i = 0; i < da.size(); ++i) {
    diff = static_cast<uint8_t>(diff | (da[i] ^ db[i]));
  }
  return diff == 0;
}

std::string random_hex(size_t n_bytes) {
  std::vector<uint8_t> buf(n_bytes ? n_bytes : 16);
  std::ifstream urandom("/dev/urandom", std::ios::binary);
  if (urandom) {
    urandom.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  }
  if (!urandom || static_cast<size_t>(urandom.gcount()) != buf.size()) {
    // Never silently hand back a weak secret.
    return {};
  }
  static const char* kHex = "0123456789abcdef";
  std::string out(buf.size() * 2, '0');
  for (size_t i = 0; i < buf.size(); ++i) {
    out[i * 2] = kHex[buf[i] >> 4];
    out[i * 2 + 1] = kHex[buf[i] & 0x0F];
  }
  return out;
}

Status IntegrityManifest::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return Status::fail(Errc::not_found, "cannot open manifest " + path);
  }
  entries_.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ls(line);
    IntegrityEntry e;
    if (!(ls >> e.sha256 >> e.path)) {
      continue;
    }
    ls >> e.bytes;
    entries_.push_back(std::move(e));
  }
  return Status::OK();
}

Status IntegrityManifest::save(const std::string& path) const {
  std::ofstream out(path);
  if (!out) {
    return Status::fail(Errc::io, "cannot write manifest " + path);
  }
  out << "# oracle model integrity manifest: <sha256>  <path>  <bytes>\n";
  for (const auto& e : entries_) {
    out << e.sha256 << "  " << e.path << "  " << e.bytes << "\n";
  }
  return Status::OK();
}

void IntegrityManifest::record(const IntegrityEntry& e) {
  for (auto& x : entries_) {
    if (x.path == e.path) {
      x = e;
      return;
    }
  }
  entries_.push_back(e);
}

const IntegrityEntry* IntegrityManifest::find(const std::string& path) const {
  for (const auto& e : entries_) {
    if (e.path == path) {
      return &e;
    }
  }
  return nullptr;
}

Status IntegrityManifest::verify(const std::string& path) const {
  const auto* e = find(path);
  if (!e) {
    return Status::fail(Errc::not_found, "no manifest entry for " + path);
  }
  std::string hex;
  uint64_t bytes = 0;
  auto st = sha256_file(path, &hex, &bytes);
  if (!st) {
    return st;
  }
  if (!constant_time_equals(hex, e->sha256)) {
    return Status::fail(Errc::protocol,
                        "integrity check failed for " + path + ": expected " + e->sha256 + ", got " + hex);
  }
  if (e->bytes && e->bytes != bytes) {
    return Status::fail(Errc::protocol, "size mismatch for " + path);
  }
  return Status::OK();
}

}  // namespace oracle::security

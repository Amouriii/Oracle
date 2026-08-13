#include "oracle/types.hpp"

#include <array>

namespace oracle {
namespace {

constexpr std::array<uint32_t, 256> kCrcTable = [] {
  std::array<uint32_t, 256> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int j = 0; j < 8; ++j) {
      c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
    t[i] = c;
  }
  return t;
}();

}  // namespace

uint32_t crc32(std::span<const std::byte> data) noexcept {
  uint32_t c = 0xFFFFFFFFu;
  for (auto b : data) {
    c = kCrcTable[(c ^ static_cast<uint8_t>(b)) & 0xFFu] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFFu;
}

Status encode_header(const TensorHeader& hdr, std::span<std::byte, sizeof(TensorHeader)> out) {
  if (!hdr.valid_magic()) {
    return Status::fail(Errc::protocol, "encode: bad magic");
  }
  if (hdr.version != kProtocolVersion) {
    return Status::fail(Errc::protocol, "encode: unsupported version");
  }
  std::memcpy(out.data(), &hdr, sizeof(TensorHeader));
  return Status::OK();
}

Status decode_header(std::span<const std::byte> in, TensorHeader* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "decode: null out");
  }
  if (in.size() < sizeof(TensorHeader)) {
    return Status::fail(Errc::protocol, "decode: truncated header");
  }
  TensorHeader hdr{};
  std::memcpy(&hdr, in.data(), sizeof(TensorHeader));
  if (!hdr.valid_magic()) {
    return Status::fail(Errc::protocol, "decode: magic != ORCL");
  }
  if (hdr.version != kProtocolVersion) {
    return Status::fail(Errc::protocol, "decode: version mismatch");
  }
  if (hdr.rank > 8) {
    return Status::fail(Errc::protocol, "decode: rank > 8");
  }
  *out = hdr;
  return Status::OK();
}

uint64_t tensor_nbytes_from_shape(DType dtype, uint32_t rank, const uint32_t shape[8]) {
  if (rank == 0) {
    return 0;
  }
  uint64_t n = dtype_size(dtype);
  for (uint32_t i = 0; i < rank && i < 8; ++i) {
    n *= static_cast<uint64_t>(shape[i]);
  }
  return n;
}

}  // namespace oracle

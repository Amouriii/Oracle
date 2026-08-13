#include "oracle/types.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
  oracle::TensorHeader hdr;
  assert(hdr.valid_magic());
  assert(sizeof(oracle::TensorHeader) == 76);
  hdr.dtype = static_cast<uint16_t>(oracle::DType::F16);
  hdr.rank = 2;
  hdr.shape[0] = 1;
  hdr.shape[1] = 8192;
  hdr.nbytes = oracle::tensor_nbytes_from_shape(oracle::DType::F16, 2, hdr.shape);
  assert(hdr.nbytes == 16384);
  hdr.flags = oracle::kFlagDecode | oracle::kFlagLastStageLogits;
  assert(hdr.is_decode());
  assert(hdr.is_logits());

  std::byte buf[sizeof(oracle::TensorHeader)];
  auto st = oracle::encode_header(hdr, std::span<std::byte, sizeof(oracle::TensorHeader)>(buf, sizeof(buf)));
  assert(st.ok());

  oracle::TensorHeader decoded{};
  st = oracle::decode_header(std::span<const std::byte>(buf, sizeof(buf)), &decoded);
  assert(st.ok());
  assert(decoded.nbytes == 16384);
  assert(decoded.shape[1] == 8192);

  std::vector<std::byte> payload(16, std::byte{0xAB});
  const uint32_t c = oracle::crc32(payload);
  assert(c != 0);

  decoded.magic = 0;
  std::byte bad[sizeof(oracle::TensorHeader)];
  std::memcpy(bad, &decoded, sizeof(decoded));
  oracle::TensorHeader tmp{};
  st = oracle::decode_header(std::span<const std::byte>(bad, sizeof(bad)), &tmp);
  assert(!st.ok());
  std::cout << "test_protocol ok magic=ORCL size=" << sizeof(oracle::TensorHeader) << "\n";
  return 0;
}

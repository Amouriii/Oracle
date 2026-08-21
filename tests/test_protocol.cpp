#include "oracle/types.hpp"

#include "check.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
  oracle::TensorHeader hdr;
  CHECK(hdr.valid_magic());
  CHECK(sizeof(oracle::TensorHeader) == 76);
  hdr.dtype = static_cast<uint16_t>(oracle::DType::F16);
  hdr.rank = 2;
  hdr.shape[0] = 1;
  hdr.shape[1] = 8192;
  hdr.nbytes = oracle::tensor_nbytes_from_shape(oracle::DType::F16, 2, hdr.shape);
  CHECK(hdr.nbytes == 16384);
  hdr.flags = oracle::kFlagDecode | oracle::kFlagLastStageLogits;
  CHECK(hdr.is_decode());
  CHECK(hdr.is_logits());

  std::byte buf[sizeof(oracle::TensorHeader)];
  auto st = oracle::encode_header(hdr, std::span<std::byte, sizeof(oracle::TensorHeader)>(buf, sizeof(buf)));
  CHECK(st.ok());

  oracle::TensorHeader decoded{};
  st = oracle::decode_header(std::span<const std::byte>(buf, sizeof(buf)), &decoded);
  CHECK(st.ok());
  CHECK(decoded.nbytes == 16384);
  CHECK(decoded.shape[1] == 8192);

  std::vector<std::byte> payload(16, std::byte{0xAB});
  const uint32_t c = oracle::crc32(payload);
  CHECK(c != 0);

  decoded.magic = 0;
  std::byte bad[sizeof(oracle::TensorHeader)];
  std::memcpy(bad, &decoded, sizeof(decoded));
  oracle::TensorHeader tmp{};
  st = oracle::decode_header(std::span<const std::byte>(bad, sizeof(bad)), &tmp);
  CHECK(!st.ok());
  std::cout << "test_protocol ok magic=ORCL size=" << sizeof(oracle::TensorHeader) << "\n";
  return 0;
}

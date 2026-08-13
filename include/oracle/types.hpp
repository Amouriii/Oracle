#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oracle {

inline constexpr uint32_t kTensorMagic = 0x4C43524Fu;  // 'ORCL' little-endian
inline constexpr uint16_t kProtocolVersion = 1;

enum class DType : uint16_t {
  F32 = 1,
  F16 = 2,
  I8 = 3,
  I32 = 4,
};

enum class Errc : int32_t {
  ok = 0,
  invalid_argument = 1,
  io = 2,
  timeout = 3,
  disconnected = 4,
  oom = 5,
  pressure = 6,
  protocol = 7,
  not_found = 8,
  not_implemented = 9,
  worker_dead = 10,
  busy = 11,
};

enum TensorFlags : uint32_t {
  kFlagPrefill = 1u << 0,
  kFlagDecode = 1u << 1,
  kFlagLastStageLogits = 1u << 2,
  kFlagHeartbeat = 1u << 3,
  kFlagLayerBlob = 1u << 4,
  kFlagControl = 1u << 5,
  kFlagEos = 1u << 6,
  kFlagAck = 1u << 7,
};

using NodeId = uint32_t;

struct Status {
  Errc code{Errc::ok};
  std::string message;

  [[nodiscard]] bool ok() const noexcept { return code == Errc::ok; }
  explicit operator bool() const noexcept { return ok(); }

  static Status OK() { return {}; }
  static Status fail(Errc c, std::string m) { return Status{c, std::move(m)}; }
};

[[nodiscard]] inline std::string_view errc_name(Errc c) {
  switch (c) {
    case Errc::ok:
      return "ok";
    case Errc::invalid_argument:
      return "invalid_argument";
    case Errc::io:
      return "io";
    case Errc::timeout:
      return "timeout";
    case Errc::disconnected:
      return "disconnected";
    case Errc::oom:
      return "oom";
    case Errc::pressure:
      return "pressure";
    case Errc::protocol:
      return "protocol";
    case Errc::not_found:
      return "not_found";
    case Errc::not_implemented:
      return "not_implemented";
    case Errc::worker_dead:
      return "worker_dead";
    case Errc::busy:
      return "busy";
  }
  return "unknown";
}

[[nodiscard]] inline uint32_t dtype_size(DType d) {
  switch (d) {
    case DType::F32:
      return 4;
    case DType::F16:
      return 2;
    case DType::I8:
      return 1;
    case DType::I32:
      return 4;
  }
  return 0;
}

[[nodiscard]] inline const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32:
      return "f32";
    case DType::F16:
      return "f16";
    case DType::I8:
      return "i8";
    case DType::I32:
      return "i32";
  }
  return "unknown";
}

#pragma pack(push, 1)
struct TensorHeader {
  uint32_t magic{kTensorMagic};
  uint16_t version{kProtocolVersion};
  uint16_t dtype{static_cast<uint16_t>(DType::F16)};
  uint32_t rank{0};
  uint32_t shape[8]{};
  uint64_t nbytes{0};
  uint64_t seq_id{0};
  uint32_t layer_id{0};
  uint32_t token_id{0};
  uint32_t flags{0};
  uint32_t checksum{0};  // CRC32 of payload; 0 means unchecked

  [[nodiscard]] DType dtype_enum() const noexcept { return static_cast<DType>(dtype); }
  [[nodiscard]] bool valid_magic() const noexcept { return magic == kTensorMagic; }
  [[nodiscard]] bool is_prefill() const noexcept { return (flags & kFlagPrefill) != 0; }
  [[nodiscard]] bool is_decode() const noexcept { return (flags & kFlagDecode) != 0; }
  [[nodiscard]] bool is_logits() const noexcept { return (flags & kFlagLastStageLogits) != 0; }
};
#pragma pack(pop)

static_assert(sizeof(TensorHeader) == 76, "TensorHeader wire size must stay packed at 76 bytes");
static_assert(alignof(TensorHeader) == 1, "TensorHeader must be packed");

struct Tensor {
  TensorHeader header{};
  std::vector<std::byte> payload;

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return payload; }
  [[nodiscard]] std::span<std::byte> bytes() noexcept { return payload; }
};

struct LayerRange {
  uint32_t start{0};
  uint32_t end{0};  // exclusive
  [[nodiscard]] uint32_t count() const noexcept { return end > start ? end - start : 0; }
};

struct ModelMeta {
  std::string name;
  std::string path;
  uint32_t n_layers{80};
  uint32_t hidden_dim{8192};
  uint32_t n_heads{64};
  uint32_t n_kv_heads{8};
  uint32_t head_dim{128};
  uint32_t n_vocab{32000};
  uint32_t max_seq{4096};
  DType weight_dtype{DType::I8};
  DType act_dtype{DType::F16};
  uint64_t bytes_per_layer{0};
  uint64_t total_weight_bytes{0};
};

uint32_t crc32(std::span<const std::byte> data) noexcept;

Status encode_header(const TensorHeader& hdr, std::span<std::byte, sizeof(TensorHeader)> out);
Status decode_header(std::span<const std::byte> in, TensorHeader* out);
uint64_t tensor_nbytes_from_shape(DType dtype, uint32_t rank, const uint32_t shape[8]);

}  // namespace oracle

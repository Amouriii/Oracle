#pragma once

// Header-only IEEE half / bfloat16 conversions.  Kept in the compute layer so
// both the BLAS shim and the dequantisers can use them without a dependency
// cycle between the two libraries.

#include <cstdint>
#include <cstring>

namespace oracle::compute {

inline float fp16_to_fp32(uint16_t h) noexcept {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      // Subnormal half: renormalise into a normal float.
      uint32_t e = 127 - 15 + 1;
      uint32_t m = man;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      f = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

inline uint16_t fp32_to_fp16(float value) noexcept {
  uint32_t f;
  std::memcpy(&f, &value, 4);
  const uint32_t sign = (f >> 16) & 0x8000u;
  const uint32_t raw_exp = (f >> 23) & 0xFFu;
  const uint32_t man = f & 0x7FFFFFu;
  if (raw_exp == 0xFFu) {
    return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
  }
  const int32_t exp = static_cast<int32_t>(raw_exp) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    // Subnormal half: shift the implicit leading 1 back in.
    const uint32_t m = (man | 0x800000u) >> static_cast<uint32_t>(1 - exp + 13);
    return static_cast<uint16_t>(sign | m);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13);
  if ((man & 0x1FFFu) > 0x1000u) {
    ++h;  // round to nearest
  }
  return static_cast<uint16_t>(h);
}

inline float bf16_to_fp32(uint16_t h) noexcept {
  const uint32_t f = static_cast<uint32_t>(h) << 16;
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

inline uint16_t fp32_to_bf16(float value) noexcept {
  uint32_t f;
  std::memcpy(&f, &value, 4);
  const uint32_t rounded = f + 0x7FFFu + ((f >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}

}  // namespace oracle::compute

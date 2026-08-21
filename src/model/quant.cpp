#include "oracle/model/quant.hpp"

#include "oracle/model/gguf.hpp"

#include <cstring>

namespace oracle::model {
namespace {

constexpr int QK_K = 256;
constexpr int K_SCALE_SIZE = 12;

inline float half(uint16_t h) { return fp16_to_fp32(h); }

inline uint16_t rd_u16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

inline uint32_t rd_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

// Q4_K / Q5_K pack eight 6-bit scale/min pairs into twelve bytes.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

void deq_q4_0(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = half(rd_u16(p));
    const uint8_t* qs = p + 2;
    for (int j = 0; j < 16; ++j) {
      y[j] = static_cast<float>((qs[j] & 0x0F) - 8) * d;
      y[j + 16] = static_cast<float>((qs[j] >> 4) - 8) * d;
    }
    p += 18;
    y += 32;
  }
}

void deq_q4_1(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = half(rd_u16(p));
    const float m = half(rd_u16(p + 2));
    const uint8_t* qs = p + 4;
    for (int j = 0; j < 16; ++j) {
      y[j] = static_cast<float>(qs[j] & 0x0F) * d + m;
      y[j + 16] = static_cast<float>(qs[j] >> 4) * d + m;
    }
    p += 20;
    y += 32;
  }
}

void deq_q5_0(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = half(rd_u16(p));
    const uint32_t qh = rd_u32(p + 2);
    const uint8_t* qs = p + 6;
    for (int j = 0; j < 16; ++j) {
      const uint8_t xh0 = static_cast<uint8_t>(((qh >> j) << 4) & 0x10);
      const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
      y[j] = static_cast<float>(static_cast<int>((qs[j] & 0x0F) | xh0) - 16) * d;
      y[j + 16] = static_cast<float>(static_cast<int>((qs[j] >> 4) | xh1) - 16) * d;
    }
    p += 22;
    y += 32;
  }
}

void deq_q5_1(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = half(rd_u16(p));
    const float m = half(rd_u16(p + 2));
    const uint32_t qh = rd_u32(p + 4);
    const uint8_t* qs = p + 8;
    for (int j = 0; j < 16; ++j) {
      const uint8_t xh0 = static_cast<uint8_t>(((qh >> j) << 4) & 0x10);
      const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
      y[j] = static_cast<float>((qs[j] & 0x0F) | xh0) * d + m;
      y[j + 16] = static_cast<float>((qs[j] >> 4) | xh1) * d + m;
    }
    p += 24;
    y += 32;
  }
}

void deq_q8_0(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = half(rd_u16(p));
    const auto* qs = reinterpret_cast<const int8_t*>(p + 2);
    for (int j = 0; j < 32; ++j) {
      y[j] = static_cast<float>(qs[j]) * d;
    }
    p += 34;
    y += 32;
  }
}

void deq_q2_k(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* scales = p;
    const uint8_t* q = p + QK_K / 16;
    const float d = half(rd_u16(p + QK_K / 16 + QK_K / 4));
    const float dmin = half(rd_u16(p + QK_K / 16 + QK_K / 4 + 2));
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = scales[is++];
        float dl = d * static_cast<float>(sc & 0x0F);
        float ml = dmin * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
        }
        sc = scales[is++];
        dl = d * static_cast<float>(sc & 0x0F);
        ml = dmin * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
        }
        shift += 2;
      }
      q += 32;
    }
    p += 84;
  }
}

void deq_q3_k(const uint8_t* p, float* y, int64_t nb) {
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* hm = p;
    const uint8_t* q = p + QK_K / 8;
    const uint8_t* sc_raw = p + QK_K / 8 + QK_K / 4;
    const float d_all = half(rd_u16(p + QK_K / 8 + QK_K / 4 + 12));

    uint32_t aux[4];
    std::memcpy(aux, sc_raw, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const auto* scales = reinterpret_cast<const int8_t*>(aux);

    uint8_t m = 1;
    int is = 0;
    const uint8_t* qq = q;
    for (int n = 0; n < QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>(static_cast<int>((qq[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        }
        dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl *
                 static_cast<float>(static_cast<int>((qq[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        }
        shift += 2;
        m <<= 1;
      }
      qq += 32;
    }
    p += 110;
  }
}

void deq_q4_k(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = half(rd_u16(p));
    const float dmin = half(rd_u16(p + 2));
    const uint8_t* scales = p + 4;
    const uint8_t* q = p + 4 + K_SCALE_SIZE;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
      uint8_t sc = 0, m = 0;
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = dmin * static_cast<float>(m);
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = dmin * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) {
        *y++ = d1 * static_cast<float>(q[l] & 0x0F) - m1;
      }
      for (int l = 0; l < 32; ++l) {
        *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
      }
      q += 32;
      is += 2;
    }
    p += 144;
  }
}

void deq_q5_k(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = half(rd_u16(p));
    const float dmin = half(rd_u16(p + 2));
    const uint8_t* scales = p + 4;
    const uint8_t* qh = p + 4 + K_SCALE_SIZE;
    const uint8_t* ql = qh + QK_K / 8;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
      uint8_t sc = 0, m = 0;
      get_scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = dmin * static_cast<float>(m);
      get_scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = dmin * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) {
        *y++ = d1 * static_cast<float>((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
      }
      for (int l = 0; l < 32; ++l) {
        *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      }
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
    p += 176;
  }
}

void deq_q6_k(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* ql = p;
    const uint8_t* qh = p + QK_K / 2;
    const auto* sc = reinterpret_cast<const int8_t*>(p + QK_K / 2 + QK_K / 4);
    const float d = half(rd_u16(p + QK_K / 2 + QK_K / 4 + QK_K / 16));
    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = static_cast<int>((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
    p += 210;
  }
}

void deq_q8_k(const uint8_t* p, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    float d;
    std::memcpy(&d, p, 4);
    const auto* qs = reinterpret_cast<const int8_t*>(p + 4);
    for (int j = 0; j < QK_K; ++j) {
      y[j] = d * static_cast<float>(qs[j]);
    }
    p += 4 + QK_K + QK_K / 16 * 2;
    y += QK_K;
  }
}

}  // namespace

void fp16_to_fp32_row(const uint16_t* src, float* dst, int64_t n) noexcept {
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = fp16_to_fp32(src[i]);
  }
}

void fp32_to_fp16_row(const float* src, uint16_t* dst, int64_t n) noexcept {
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = fp32_to_fp16(src[i]);
  }
}

bool dequantize_supported(uint32_t t) noexcept { return ggml_type_info(t).dequantizable; }

Status dequantize_row(uint32_t type, const void* src, float* dst, int64_t n) {
  if (!src || !dst) {
    return Status::fail(Errc::invalid_argument, "dequantize: null buffer");
  }
  if (n <= 0) {
    return Status::OK();
  }
  const auto& info = ggml_type_info(type);
  if (!info.dequantizable) {
    return Status::fail(Errc::not_implemented,
                        std::string("dequantize: unsupported ggml type ") + info.name);
  }
  if (info.block_size == 0 || (n % info.block_size) != 0) {
    return Status::fail(Errc::invalid_argument,
                        std::string("dequantize: ") + std::to_string(n) + " is not a multiple of the " +
                            info.name + " block size");
  }
  const auto* p = static_cast<const uint8_t*>(src);
  const int64_t nb = n / info.block_size;
  switch (type) {
    case GGML_F32:
      std::memcpy(dst, p, static_cast<size_t>(n) * 4);
      return Status::OK();
    case GGML_F16:
      for (int64_t i = 0; i < n; ++i) {
        dst[i] = fp16_to_fp32(rd_u16(p + i * 2));
      }
      return Status::OK();
    case GGML_BF16:
      for (int64_t i = 0; i < n; ++i) {
        dst[i] = bf16_to_fp32(rd_u16(p + i * 2));
      }
      return Status::OK();
    case GGML_Q4_0:
      deq_q4_0(p, dst, nb);
      return Status::OK();
    case GGML_Q4_1:
      deq_q4_1(p, dst, nb);
      return Status::OK();
    case GGML_Q5_0:
      deq_q5_0(p, dst, nb);
      return Status::OK();
    case GGML_Q5_1:
      deq_q5_1(p, dst, nb);
      return Status::OK();
    case GGML_Q8_0:
      deq_q8_0(p, dst, nb);
      return Status::OK();
    case GGML_Q2_K:
      deq_q2_k(p, dst, nb);
      return Status::OK();
    case GGML_Q3_K:
      deq_q3_k(p, dst, nb);
      return Status::OK();
    case GGML_Q4_K:
      deq_q4_k(p, dst, nb);
      return Status::OK();
    case GGML_Q5_K:
      deq_q5_k(p, dst, nb);
      return Status::OK();
    case GGML_Q6_K:
      deq_q6_k(p, dst, nb);
      return Status::OK();
    case GGML_Q8_K:
      deq_q8_k(p, dst, nb);
      return Status::OK();
    default:
      break;
  }
  return Status::fail(Errc::not_implemented, std::string("dequantize: no kernel for ") + info.name);
}

}  // namespace oracle::model

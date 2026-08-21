// Checks each block format against hand-built blocks whose expansion is known
// exactly from the ggml layout, plus the f16/bf16 conversions.
#include "oracle/model/gguf.hpp"
#include "oracle/model/quant.hpp"

#include "check.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace oracle;
using namespace oracle::model;

namespace {

void put16(std::vector<uint8_t>& b, size_t off, uint16_t v) { std::memcpy(b.data() + off, &v, 2); }

bool close(float a, float b, float eps = 1e-4f) { return std::abs(a - b) <= eps; }

void check_type_sizes() {
  CHECK(ggml_type_info(GGML_F32).type_size == 4);
  CHECK(ggml_type_info(GGML_Q4_0).type_size == 18);
  CHECK(ggml_type_info(GGML_Q4_1).type_size == 20);
  CHECK(ggml_type_info(GGML_Q5_0).type_size == 22);
  CHECK(ggml_type_info(GGML_Q5_1).type_size == 24);
  CHECK(ggml_type_info(GGML_Q8_0).type_size == 34);
  CHECK(ggml_type_info(GGML_Q2_K).type_size == 84);
  CHECK(ggml_type_info(GGML_Q3_K).type_size == 110);
  CHECK(ggml_type_info(GGML_Q4_K).type_size == 144);
  CHECK(ggml_type_info(GGML_Q5_K).type_size == 176);
  CHECK(ggml_type_info(GGML_Q6_K).type_size == 210);
  CHECK(ggml_row_bytes(GGML_Q4_K, 512) == 288);
  CHECK(ggml_row_bytes(GGML_F16, 100) == 200);
  CHECK(std::abs(ggml_bits_per_weight(GGML_Q4_0) - 4.5) < 1e-9);
  CHECK(std::abs(ggml_bits_per_weight(GGML_Q8_0) - 8.5) < 1e-9);
  // An unknown type must not claim a size.
  CHECK(ggml_type_info(9999).block_size == 0);
  CHECK(!dequantize_supported(9999));
}

void check_halves() {
  const float samples[] = {0.f, 1.f, -1.f, 0.5f, -2.25f, 65504.f, 1e-5f, 3.14159f};
  for (float v : samples) {
    CHECK(close(fp16_to_fp32(fp32_to_fp16(v)), v, std::abs(v) * 1e-3f + 1e-6f));
    CHECK(close(bf16_to_fp32(fp32_to_bf16(v)), v, std::abs(v) * 1e-2f + 1e-6f));
  }
  CHECK(fp16_to_fp32(0) == 0.f);
  CHECK(std::isinf(fp16_to_fp32(0x7C00)));
  CHECK(std::isnan(fp16_to_fp32(0x7E00)));
  // Subnormal halves must not flush to zero.
  CHECK(fp16_to_fp32(1) > 0.f && fp16_to_fp32(1) < 1e-6f);
}

void check_q8_0() {
  std::vector<uint8_t> b(34);
  put16(b, 0, fp32_to_fp16(0.5f));
  for (int j = 0; j < 32; ++j) {
    b[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(j - 16));
  }
  float y[32];
  CHECK(dequantize_row(GGML_Q8_0, b.data(), y, 32).ok());
  for (int j = 0; j < 32; ++j) {
    CHECK(close(y[j], 0.5f * static_cast<float>(j - 16)));
  }
}

void check_q4_0() {
  std::vector<uint8_t> b(18);
  put16(b, 0, fp32_to_fp16(2.0f));
  for (int j = 0; j < 16; ++j) {
    b[2 + j] = static_cast<uint8_t>(j);  // low nibble j, high nibble 0
  }
  float y[32];
  CHECK(dequantize_row(GGML_Q4_0, b.data(), y, 32).ok());
  for (int j = 0; j < 16; ++j) {
    CHECK(close(y[j], 2.0f * static_cast<float>(j - 8)));
    CHECK(close(y[j + 16], -16.0f));
  }
}

void check_q4_1() {
  std::vector<uint8_t> b(20);
  put16(b, 0, fp32_to_fp16(2.0f));
  put16(b, 2, fp32_to_fp16(1.0f));
  for (int j = 0; j < 16; ++j) {
    b[4 + j] = static_cast<uint8_t>(j);
  }
  float y[32];
  CHECK(dequantize_row(GGML_Q4_1, b.data(), y, 32).ok());
  for (int j = 0; j < 16; ++j) {
    CHECK(close(y[j], 2.0f * static_cast<float>(j) + 1.0f));
    CHECK(close(y[j + 16], 1.0f));
  }
}

void check_q5_0() {
  std::vector<uint8_t> b(22);
  put16(b, 0, fp32_to_fp16(1.0f));
  std::memset(b.data() + 2, 0, 4);  // qh: no high bits set
  for (int j = 0; j < 16; ++j) {
    b[6 + j] = static_cast<uint8_t>(j);
  }
  float y[32];
  CHECK(dequantize_row(GGML_Q5_0, b.data(), y, 32).ok());
  for (int j = 0; j < 16; ++j) {
    CHECK(close(y[j], static_cast<float>(j - 16)));
    CHECK(close(y[j + 16], -16.0f));
  }
  // Setting every qh bit lifts each quant by 16.
  std::memset(b.data() + 2, 0xFF, 4);
  CHECK(dequantize_row(GGML_Q5_0, b.data(), y, 32).ok());
  for (int j = 0; j < 16; ++j) {
    CHECK(close(y[j], static_cast<float>(j - 16 + 16)));
  }
}

void check_q4_k() {
  // scales bytes all 1 => sc == 1 for every sub-block, min == 1 for the first
  // four and 0 for the last four (see get_scale_min_k4).
  std::vector<uint8_t> b(144, 0);
  put16(b, 0, fp32_to_fp16(1.0f));
  put16(b, 2, fp32_to_fp16(1.0f));
  for (int i = 0; i < 12; ++i) {
    b[4 + i] = 1;
  }
  std::vector<float> y(256);
  CHECK(dequantize_row(GGML_Q4_K, b.data(), y.data(), 256).ok());
  for (int i = 0; i < 128; ++i) {
    CHECK(close(y[i], -1.0f));
  }
  for (int i = 128; i < 256; ++i) {
    CHECK(close(y[i], 0.0f));
  }
}

void check_q5_k() {
  std::vector<uint8_t> b(176, 0);
  put16(b, 0, fp32_to_fp16(1.0f));
  put16(b, 2, fp32_to_fp16(1.0f));
  for (int i = 0; i < 12; ++i) {
    b[4 + i] = 1;
  }
  std::vector<float> y(256);
  CHECK(dequantize_row(GGML_Q5_K, b.data(), y.data(), 256).ok());
  for (int i = 0; i < 128; ++i) {
    CHECK(close(y[i], -1.0f));
  }
  for (int i = 128; i < 256; ++i) {
    CHECK(close(y[i], 0.0f));
  }
}

void check_q6_k() {
  std::vector<uint8_t> b(210, 0);
  for (int i = 0; i < 16; ++i) {
    b[128 + 64 + i] = 1;  // int8 scales
  }
  put16(b, 128 + 64 + 16, fp32_to_fp16(1.0f));
  std::vector<float> y(256);
  CHECK(dequantize_row(GGML_Q6_K, b.data(), y.data(), 256).ok());
  for (int i = 0; i < 256; ++i) {
    CHECK(close(y[i], -32.0f));  // zero quants sit at the -32 offset
  }
}

void check_q2_k() {
  std::vector<uint8_t> b(84, 0);
  for (int i = 0; i < 16; ++i) {
    b[i] = 0x11;  // scale 1, min 1
  }
  put16(b, 16 + 64, fp32_to_fp16(1.0f));
  put16(b, 16 + 64 + 2, fp32_to_fp16(1.0f));
  std::vector<float> y(256);
  CHECK(dequantize_row(GGML_Q2_K, b.data(), y.data(), 256).ok());
  for (int i = 0; i < 256; ++i) {
    CHECK(close(y[i], -1.0f));
  }
}

void check_q3_k() {
  // All-zero scale bytes decode to scale 0, i.e. dl = d * (0 - 32); zero quant
  // bits with a clear hmask decode to (0 - 4).
  std::vector<uint8_t> b(110, 0);
  put16(b, 32 + 64 + 12, fp32_to_fp16(1.0f));
  std::vector<float> y(256);
  CHECK(dequantize_row(GGML_Q3_K, b.data(), y.data(), 256).ok());
  for (int i = 0; i < 256; ++i) {
    CHECK(close(y[i], 128.0f));
  }
}

void check_errors() {
  std::vector<uint8_t> b(144, 0);
  float y[4];
  // A count that is not a whole number of blocks must be rejected.
  CHECK(dequantize_row(GGML_Q4_K, b.data(), y, 4).code == Errc::invalid_argument);
  // Types we deliberately do not implement report not_implemented, not garbage.
  std::vector<float> big(256);
  CHECK(dequantize_row(GGML_IQ2_XXS, b.data(), big.data(), 256).code == Errc::not_implemented);
  CHECK(dequantize_row(GGML_F32, nullptr, y, 4).code == Errc::invalid_argument);
}

}  // namespace

int main() {
  check_type_sizes();
  check_halves();
  check_q8_0();
  check_q4_0();
  check_q4_1();
  check_q5_0();
  check_q4_k();
  check_q5_k();
  check_q6_k();
  check_q2_k();
  check_q3_k();
  check_errors();
  std::cout << "test_quant ok\n";
  return 0;
}

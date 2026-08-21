#pragma once

// Dequantisation of ggml block formats into f32.
//
// Oracle keeps weights mapped in their quantised form and expands one row at a
// time inside the mat-vec inner loop, so a 4-bit 8B model stays ~4.5 GB resident
// instead of the ~32 GB an up-front f32 expansion would need.

#include "oracle/compute/half.hpp"
#include "oracle/types.hpp"

#include <cstdint>

namespace oracle::model {

inline float fp16_to_fp32(uint16_t h) noexcept { return compute::fp16_to_fp32(h); }
inline uint16_t fp32_to_fp16(float f) noexcept { return compute::fp32_to_fp16(f); }
inline float bf16_to_fp32(uint16_t h) noexcept { return compute::bf16_to_fp32(h); }
inline uint16_t fp32_to_bf16(float f) noexcept { return compute::fp32_to_bf16(f); }

void fp16_to_fp32_row(const uint16_t* src, float* dst, int64_t n) noexcept;
void fp32_to_fp16_row(const float* src, uint16_t* dst, int64_t n) noexcept;

[[nodiscard]] bool dequantize_supported(uint32_t ggml_type) noexcept;

// Expand `n` contiguous elements of `ggml_type` starting at `src` into `dst`.
// `n` must be a multiple of the type's block size.
Status dequantize_row(uint32_t ggml_type, const void* src, float* dst, int64_t n);

}  // namespace oracle::model

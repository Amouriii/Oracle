#include <metal_stdlib>
using namespace metal;

kernel void sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint row = gid.y;
  const uint col = gid.x;
  if (row >= M || col >= N) {
    return;
  }
  float acc = 0.0f;
  for (uint k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}

kernel void saxpy_inplace(
    device float* x [[buffer(0)]],
    constant float& a [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  x[gid] *= a;
}

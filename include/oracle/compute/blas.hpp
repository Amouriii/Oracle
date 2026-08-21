#pragma once

// Portable BLAS-ish shim.  On Apple we forward to Accelerate's cblas; everywhere
// else we use a small blocked implementation.  Only the handful of kernels the
// engine actually needs are exposed, which keeps the dependency surface at zero
// on Linux (Docker) while still using the vendor library on macOS.

#include <cstddef>
#include <cstdint>
#include <functional>

namespace oracle::compute {

// Split [0, total) across the compute pool and run `fn(begin, end)` on each
// slice, returning once every slice has finished.  The calling thread takes one
// of the slices, so a single-threaded build never allocates a thread.
void parallel_for(int total, const std::function<void(int begin, int end)>& fn);

// C(MxN) = alpha * A(MxK) * B(KxN) + beta * C, all row-major.
void sgemm_nn(int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb,
              float beta, float* c, int ldc);

// C(MxN) = alpha * A(MxK) * B(NxK)^T + beta * C, all row-major.
// This is the natural layout for weight matrices stored as [out_features, in_features].
void sgemm_nt(int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb,
              float beta, float* c, int ldc);

// y(N) = W(NxK) * x(K), W row-major [out, in].  Multi-threaded over rows.
void matvec(int n, int k, const float* w, const float* x, float* y);

float dot(const float* a, const float* b, int n);

// Dot product of an f16 vector with an f32 vector, without materialising the
// expanded f16 side.  Used for KV-cache reads during attention.
float dot_f16(const uint16_t* a, const float* b, int n);

// Number of worker threads the compute layer will use (>= 1).
int thread_count();
void set_thread_count(int n);

// Backend name for /cluster and startup banners.
const char* backend_name();

}  // namespace oracle::compute

#include "oracle/compute/blas.hpp"

#include "oracle/compute/half.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(ORACLE_HAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace oracle::compute {
namespace {

int g_threads = 0;

int default_threads() {
  const unsigned hw = std::thread::hardware_concurrency();
  return static_cast<int>(hw ? hw : 4u);
}

// A persistent fork/join pool.  Decode calls into the compute layer once per
// projection per token, so spawning threads per call would dominate the budget.
// Slices are handed out as (worker_index, worker_count) and the calling thread
// takes slice 0, which keeps single-threaded builds allocation-free.
class Pool {
 public:
  static Pool& instance() {
    static Pool p;
    return p;
  }

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  void run(const std::function<void(int, int)>& fn) {
    if (workers_.empty()) {
      fn(0, 1);
      return;
    }
    // Only one fork/join may own the pool at a time.  A second caller (another
    // request's forward pass) runs its work inline single-threaded rather than
    // blocking, so concurrent requests still make progress instead of queueing
    // behind each other for the same worker threads.
    std::unique_lock<std::mutex> exec(exec_mu_, std::try_to_lock);
    if (!exec.owns_lock()) {
      fn(0, 1);
      return;
    }
    {
      std::lock_guard<std::mutex> g(mu_);
      fn_ = &fn;
      pending_ = static_cast<int>(workers_.size());
      ++epoch_;
    }
    cv_.notify_all();
    fn(0, size());
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return pending_ == 0; });
    fn_ = nullptr;
  }

 private:
  Pool() {
    const int n = thread_count();
    workers_.reserve(static_cast<size_t>(std::max(0, n - 1)));
    for (int i = 1; i < n; ++i) {
      workers_.emplace_back([this, i] { loop(i); });
    }
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
      ++epoch_;
    }
    cv_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  void loop(int idx) {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(int, int)>* fn = nullptr;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || epoch_ != seen; });
        if (stop_) {
          return;
        }
        seen = epoch_;
        fn = fn_;
      }
      if (fn) {
        (*fn)(idx, size());
      }
      {
        std::lock_guard<std::mutex> g(mu_);
        if (--pending_ == 0) {
          done_cv_.notify_all();
        }
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex exec_mu_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  const std::function<void(int, int)>* fn_{nullptr};
  uint64_t epoch_{0};
  int pending_{0};
  bool stop_{false};
};

// Split [0, total) into `count` near-equal contiguous slices and return slice `idx`.
inline void slice(int total, int idx, int count, int* lo, int* hi) {
  const int chunk = (total + count - 1) / count;
  *lo = std::min(total, idx * chunk);
  *hi = std::min(total, *lo + chunk);
}

}  // namespace

int thread_count() { return g_threads > 0 ? g_threads : default_threads(); }

void set_thread_count(int n) { g_threads = n > 0 ? n : 0; }

const char* backend_name() {
#if defined(ORACLE_HAS_ACCELERATE)
  return "accelerate";
#else
  return "portable-cpu";
#endif
}

void parallel_for(int total, const std::function<void(int, int)>& fn) {
  if (total <= 0) {
    return;
  }
  Pool::instance().run([&](int idx, int count) {
    const int effective = std::min(count, total);
    if (idx >= effective) {
      return;
    }
    int lo = 0, hi = 0;
    slice(total, idx, effective, &lo, &hi);
    if (lo < hi) {
      fn(lo, hi);
    }
  });
}

float dot_f16(const uint16_t* a, const float* b, int n) {
  float s0 = 0.f, s1 = 0.f;
  int i = 0;
  for (; i + 1 < n; i += 2) {
    s0 += fp16_to_fp32(a[i]) * b[i];
    s1 += fp16_to_fp32(a[i + 1]) * b[i + 1];
  }
  for (; i < n; ++i) {
    s0 += fp16_to_fp32(a[i]) * b[i];
  }
  return s0 + s1;
}

float dot(const float* a, const float* b, int n) {
  // Four accumulators so the compiler can keep the FMA pipeline busy; this is
  // the innermost loop of every projection during decode.
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) {
    s0 += a[i] * b[i];
  }
  return (s0 + s1) + (s2 + s3);
}

void matvec(int n, int k, const float* w, const float* x, float* y) {
  if (n <= 0 || k <= 0) {
    return;
  }
  Pool::instance().run([&](int idx, int count) {
    int lo = 0, hi = 0;
    slice(n, idx, count, &lo, &hi);
    for (int r = lo; r < hi; ++r) {
      y[r] = dot(w + static_cast<size_t>(r) * k, x, k);
    }
  });
}

void sgemm_nn(int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb,
              float beta, float* c, int ldc) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
#if defined(ORACLE_HAS_ACCELERATE)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
  Pool::instance().run([&](int idx, int count) {
    int lo = 0, hi = 0;
    slice(m, idx, count, &lo, &hi);
    for (int i = lo; i < hi; ++i) {
      float* crow = c + static_cast<size_t>(i) * ldc;
      if (beta == 0.f) {
        std::memset(crow, 0, sizeof(float) * static_cast<size_t>(n));
      } else if (beta != 1.f) {
        for (int j = 0; j < n; ++j) {
          crow[j] *= beta;
        }
      }
      const float* arow = a + static_cast<size_t>(i) * lda;
      for (int p = 0; p < k; ++p) {
        const float av = alpha * arow[p];
        if (av == 0.f) {
          continue;
        }
        const float* brow = b + static_cast<size_t>(p) * ldb;
        for (int j = 0; j < n; ++j) {
          crow[j] += av * brow[j];
        }
      }
    }
  });
#endif
}

void sgemm_nt(int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb,
              float beta, float* c, int ldc) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }
#if defined(ORACLE_HAS_ACCELERATE)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
  // Slice over N: the decode path calls this with m == 1 (one token) and a very
  // large n (vocab), so slicing over M would leave every worker but one idle.
  Pool::instance().run([&](int idx, int count) {
    int lo = 0, hi = 0;
    slice(n, idx, count, &lo, &hi);
    for (int i = 0; i < m; ++i) {
      const float* arow = a + static_cast<size_t>(i) * lda;
      float* crow = c + static_cast<size_t>(i) * ldc;
      for (int j = lo; j < hi; ++j) {
        const float v = alpha * dot(arow, b + static_cast<size_t>(j) * ldb, k);
        crow[j] = beta == 0.f ? v : beta * crow[j] + v;
      }
    }
  });
#endif
}

}  // namespace oracle::compute

#pragma once

// Admission control for the public HTTP surface: per-subject request rate,
// per-subject and global concurrency, request size, and a connection-rate guard
// that sheds load from a single noisy source before it reaches the model.

#include "oracle/types.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace oracle::security {

using Clock = std::chrono::steady_clock;

// Classic token bucket: `capacity` burst, refilled at `refill_per_sec`.
class TokenBucket {
 public:
  TokenBucket() = default;
  TokenBucket(double capacity, double refill_per_sec)
      : capacity_(capacity), refill_(refill_per_sec), tokens_(capacity) {}

  bool try_consume(double n, Clock::time_point now);
  [[nodiscard]] double tokens(Clock::time_point now) const;
  [[nodiscard]] double capacity() const noexcept { return capacity_; }

 private:
  void refill(Clock::time_point now) const;

  double capacity_{60};
  double refill_{1};
  mutable double tokens_{60};
  mutable Clock::time_point last_{};
  mutable bool started_{false};
};

struct RateLimitConfig {
  uint32_t requests_per_minute{120};
  uint32_t burst{20};
  // Sources exceeding this many *rejected* requests inside `ban_window` are
  // parked for `ban_seconds`, which is what stops a spray from costing a
  // signature check per packet.
  uint32_t abuse_threshold{50};
  uint32_t ban_window_seconds{60};
  uint32_t ban_seconds{300};
  uint32_t idle_evict_seconds{900};
};

// Per-subject (api key id, or client IP when unauthenticated) rate limiting.
class RateLimiter {
 public:
  explicit RateLimiter(RateLimitConfig cfg = {}) : cfg_(cfg) {}

  void configure(const RateLimitConfig& cfg);
  [[nodiscard]] const RateLimitConfig& config() const noexcept { return cfg_; }

  // `rpm_override` of 0 uses the configured default.
  [[nodiscard]] bool allow(const std::string& subject, uint32_t rpm_override = 0);
  // Records a rejection, which feeds the abuse counter.
  void penalise(const std::string& subject);
  [[nodiscard]] bool is_banned(const std::string& subject) const;
  [[nodiscard]] double retry_after_seconds(const std::string& subject) const;
  [[nodiscard]] size_t tracked_subjects() const;
  [[nodiscard]] uint64_t total_allowed() const noexcept { return allowed_.load(); }
  [[nodiscard]] uint64_t total_rejected() const noexcept { return rejected_.load(); }
  void reset();

 private:
  struct Entry {
    TokenBucket bucket;
    uint32_t strikes{0};
    Clock::time_point strike_window_start{};
    Clock::time_point banned_until{};
    Clock::time_point last_seen{};
  };

  Entry& entry_locked(const std::string& subject, uint32_t rpm_override);
  void evict_locked(Clock::time_point now);

  mutable std::mutex mu_;
  RateLimitConfig cfg_;
  std::unordered_map<std::string, Entry> entries_;
  std::atomic<uint64_t> allowed_{0};
  std::atomic<uint64_t> rejected_{0};
  Clock::time_point last_evict_{};
};

// Counts in-flight requests globally and per subject.  Acquire returns an RAII
// lease; letting it go releases the slot even on an exception or early return.
class ConcurrencyLimiter {
 public:
  class Lease {
   public:
    Lease() = default;
    Lease(ConcurrencyLimiter* owner, std::string subject) : owner_(owner), subject_(std::move(subject)) {}
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& o) noexcept { *this = std::move(o); }
    Lease& operator=(Lease&& o) noexcept;
    ~Lease();

    [[nodiscard]] bool held() const noexcept { return owner_ != nullptr; }
    void release();

   private:
    ConcurrencyLimiter* owner_{nullptr};
    std::string subject_;
  };

  void configure(uint32_t global_max, uint32_t per_subject_max);
  [[nodiscard]] Lease acquire(const std::string& subject, uint32_t per_subject_override = 0);
  [[nodiscard]] uint32_t in_flight() const;
  [[nodiscard]] uint32_t in_flight(const std::string& subject) const;
  [[nodiscard]] uint32_t global_max() const noexcept { return global_max_; }

 private:
  friend class Lease;
  void release(const std::string& subject);

  mutable std::mutex mu_;
  std::unordered_map<std::string, uint32_t> per_subject_;
  uint32_t active_{0};
  uint32_t global_max_{32};
  uint32_t per_subject_max_{8};
};

}  // namespace oracle::security

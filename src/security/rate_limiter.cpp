#include "oracle/security/rate_limiter.hpp"

#include <algorithm>

namespace oracle::security {

void TokenBucket::refill(Clock::time_point now) const {
  if (!started_) {
    last_ = now;
    started_ = true;
    return;
  }
  const double dt = std::chrono::duration<double>(now - last_).count();
  if (dt <= 0) {
    return;
  }
  last_ = now;
  tokens_ = std::min(capacity_, tokens_ + dt * refill_);
}

bool TokenBucket::try_consume(double n, Clock::time_point now) {
  refill(now);
  if (tokens_ + 1e-9 < n) {
    return false;
  }
  tokens_ -= n;
  return true;
}

double TokenBucket::tokens(Clock::time_point now) const {
  refill(now);
  return tokens_;
}

void RateLimiter::configure(const RateLimitConfig& cfg) {
  std::lock_guard<std::mutex> g(mu_);
  cfg_ = cfg;
  entries_.clear();  // old buckets were sized for the old limits
}

RateLimiter::Entry& RateLimiter::entry_locked(const std::string& subject, uint32_t rpm_override) {
  auto it = entries_.find(subject);
  if (it != entries_.end()) {
    return it->second;
  }
  const uint32_t rpm = rpm_override ? rpm_override : cfg_.requests_per_minute;
  Entry e;
  e.bucket = TokenBucket(std::max<uint32_t>(1, cfg_.burst), std::max(rpm, 1u) / 60.0);
  return entries_.emplace(subject, std::move(e)).first->second;
}

void RateLimiter::evict_locked(Clock::time_point now) {
  if (cfg_.idle_evict_seconds == 0) {
    return;
  }
  // Bound the map so a spray of distinct source addresses cannot grow it without
  // limit -- the eviction sweep itself is the memory-side DoS defence.
  if (now - last_evict_ < std::chrono::seconds(30) && entries_.size() < 100000) {
    return;
  }
  last_evict_ = now;
  const auto cutoff = std::chrono::seconds(cfg_.idle_evict_seconds);
  for (auto it = entries_.begin(); it != entries_.end();) {
    const bool idle = now - it->second.last_seen > cutoff;
    const bool banned = now < it->second.banned_until;
    it = (idle && !banned) ? entries_.erase(it) : std::next(it);
  }
}

bool RateLimiter::allow(const std::string& subject, uint32_t rpm_override) {
  const auto now = Clock::now();
  std::lock_guard<std::mutex> g(mu_);
  evict_locked(now);
  Entry& e = entry_locked(subject, rpm_override);
  e.last_seen = now;
  if (now < e.banned_until) {
    rejected_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (!e.bucket.try_consume(1.0, now)) {
    rejected_.fetch_add(1, std::memory_order_relaxed);
    if (e.strike_window_start == Clock::time_point{} ||
        now - e.strike_window_start > std::chrono::seconds(cfg_.ban_window_seconds)) {
      e.strike_window_start = now;
      e.strikes = 0;
    }
    if (++e.strikes >= cfg_.abuse_threshold && cfg_.ban_seconds) {
      e.banned_until = now + std::chrono::seconds(cfg_.ban_seconds);
    }
    return false;
  }
  allowed_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void RateLimiter::penalise(const std::string& subject) {
  const auto now = Clock::now();
  std::lock_guard<std::mutex> g(mu_);
  Entry& e = entry_locked(subject, 0);
  e.last_seen = now;
  if (e.strike_window_start == Clock::time_point{} ||
      now - e.strike_window_start > std::chrono::seconds(cfg_.ban_window_seconds)) {
    e.strike_window_start = now;
    e.strikes = 0;
  }
  if (++e.strikes >= cfg_.abuse_threshold && cfg_.ban_seconds) {
    e.banned_until = now + std::chrono::seconds(cfg_.ban_seconds);
  }
}

bool RateLimiter::is_banned(const std::string& subject) const {
  std::lock_guard<std::mutex> g(mu_);
  const auto it = entries_.find(subject);
  return it != entries_.end() && Clock::now() < it->second.banned_until;
}

double RateLimiter::retry_after_seconds(const std::string& subject) const {
  const auto now = Clock::now();
  std::lock_guard<std::mutex> g(mu_);
  const auto it = entries_.find(subject);
  if (it == entries_.end()) {
    return 0.0;
  }
  if (now < it->second.banned_until) {
    return std::chrono::duration<double>(it->second.banned_until - now).count();
  }
  const double rate = std::max(cfg_.requests_per_minute, 1u) / 60.0;
  const double have = it->second.bucket.tokens(now);
  return have >= 1.0 ? 0.0 : (1.0 - have) / rate;
}

size_t RateLimiter::tracked_subjects() const {
  std::lock_guard<std::mutex> g(mu_);
  return entries_.size();
}

void RateLimiter::reset() {
  std::lock_guard<std::mutex> g(mu_);
  entries_.clear();
  allowed_.store(0);
  rejected_.store(0);
}

ConcurrencyLimiter::Lease& ConcurrencyLimiter::Lease::operator=(Lease&& o) noexcept {
  if (this != &o) {
    release();
    owner_ = o.owner_;
    subject_ = std::move(o.subject_);
    o.owner_ = nullptr;
  }
  return *this;
}

ConcurrencyLimiter::Lease::~Lease() { release(); }

void ConcurrencyLimiter::Lease::release() {
  if (owner_) {
    owner_->release(subject_);
    owner_ = nullptr;
  }
}

void ConcurrencyLimiter::configure(uint32_t global_max, uint32_t per_subject_max) {
  std::lock_guard<std::mutex> g(mu_);
  global_max_ = global_max ? global_max : 1;
  per_subject_max_ = per_subject_max;
}

ConcurrencyLimiter::Lease ConcurrencyLimiter::acquire(const std::string& subject,
                                                      uint32_t per_subject_override) {
  std::lock_guard<std::mutex> g(mu_);
  if (active_ >= global_max_) {
    return {};
  }
  const uint32_t cap = per_subject_override ? per_subject_override : per_subject_max_;
  if (cap) {
    const auto it = per_subject_.find(subject);
    if (it != per_subject_.end() && it->second >= cap) {
      return {};
    }
  }
  ++active_;
  ++per_subject_[subject];
  return Lease(this, subject);
}

void ConcurrencyLimiter::release(const std::string& subject) {
  std::lock_guard<std::mutex> g(mu_);
  if (active_ > 0) {
    --active_;
  }
  const auto it = per_subject_.find(subject);
  if (it != per_subject_.end()) {
    if (it->second <= 1) {
      per_subject_.erase(it);
    } else {
      --it->second;
    }
  }
}

uint32_t ConcurrencyLimiter::in_flight() const {
  std::lock_guard<std::mutex> g(mu_);
  return active_;
}

uint32_t ConcurrencyLimiter::in_flight(const std::string& subject) const {
  std::lock_guard<std::mutex> g(mu_);
  const auto it = per_subject_.find(subject);
  return it == per_subject_.end() ? 0 : it->second;
}

}  // namespace oracle::security

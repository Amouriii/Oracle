#pragma once

// Structured security logging.
//
// Every admission decision, worker handshake and integrity check lands here.
// Events go to an append-only file when one is configured and always to an
// in-memory ring so /cluster and the dashboard can show recent activity without
// giving the browser filesystem access.

#include "oracle/types.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace oracle::security {

enum class Severity { Info, Notice, Warning, Alert };

const char* severity_name(Severity s);

struct SecurityEvent {
  std::chrono::system_clock::time_point at{};
  Severity severity{Severity::Info};
  std::string category;  // auth, rate_limit, worker, integrity, request
  std::string subject;   // api key id, worker id, client address
  std::string detail;

  [[nodiscard]] std::string to_json() const;
};

class AuditLog {
 public:
  AuditLog() = default;

  // Opens (or creates) an append-only log file.  Logging continues to the ring
  // buffer either way, so a read-only filesystem degrades rather than fails.
  Status open_file(const std::string& path);
  void set_ring_capacity(size_t n);
  void set_echo_to_stderr(bool on) { echo_.store(on); }

  void log(Severity sev, std::string category, std::string subject, std::string detail);
  void info(std::string cat, std::string subj, std::string detail) {
    log(Severity::Info, std::move(cat), std::move(subj), std::move(detail));
  }
  void warn(std::string cat, std::string subj, std::string detail) {
    log(Severity::Warning, std::move(cat), std::move(subj), std::move(detail));
  }
  void alert(std::string cat, std::string subj, std::string detail) {
    log(Severity::Alert, std::move(cat), std::move(subj), std::move(detail));
  }

  [[nodiscard]] std::vector<SecurityEvent> recent(size_t n = 50) const;
  [[nodiscard]] std::unordered_map<std::string, uint64_t> counters() const;
  [[nodiscard]] uint64_t count(const std::string& category) const;
  [[nodiscard]] uint64_t total() const noexcept { return total_.load(); }
  [[nodiscard]] const std::string& file_path() const noexcept { return path_; }

 private:
  mutable std::mutex mu_;
  std::deque<SecurityEvent> ring_;
  size_t ring_capacity_{256};
  std::ofstream file_;
  std::string path_;
  std::unordered_map<std::string, uint64_t> counters_;
  std::atomic<uint64_t> total_{0};
  std::atomic<bool> echo_{false};
};

}  // namespace oracle::security

#include "oracle/security/audit_log.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace oracle::security {
namespace {

std::string json_escape(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

std::string iso8601(std::chrono::system_clock::time_point tp) {
  const auto t = std::chrono::system_clock::to_time_t(tp);
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count() % 1000;
  std::tm tm{};
  gmtime_r(&t, &tm);
  std::ostringstream os;
  os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms << 'Z';
  return os.str();
}

}  // namespace

const char* severity_name(Severity s) {
  switch (s) {
    case Severity::Info: return "info";
    case Severity::Notice: return "notice";
    case Severity::Warning: return "warning";
    case Severity::Alert: return "alert";
  }
  return "info";
}

std::string SecurityEvent::to_json() const {
  std::ostringstream os;
  os << "{\"at\":\"" << iso8601(at) << "\",\"severity\":\"" << severity_name(severity)
     << "\",\"category\":\"" << json_escape(category) << "\",\"subject\":\"" << json_escape(subject)
     << "\",\"detail\":\"" << json_escape(detail) << "\"}";
  return os.str();
}

Status AuditLog::open_file(const std::string& path) {
  std::lock_guard<std::mutex> g(mu_);
  file_.close();
  file_.open(path, std::ios::app);
  if (!file_) {
    path_.clear();
    return Status::fail(Errc::io, "cannot open security log " + path);
  }
  path_ = path;
  return Status::OK();
}

void AuditLog::set_ring_capacity(size_t n) {
  std::lock_guard<std::mutex> g(mu_);
  ring_capacity_ = n ? n : 1;
  while (ring_.size() > ring_capacity_) {
    ring_.pop_front();
  }
}

void AuditLog::log(Severity sev, std::string category, std::string subject, std::string detail) {
  SecurityEvent e;
  e.at = std::chrono::system_clock::now();
  e.severity = sev;
  e.category = std::move(category);
  e.subject = std::move(subject);
  e.detail = std::move(detail);
  const std::string line = e.to_json();

  std::lock_guard<std::mutex> g(mu_);
  ++counters_[e.category];
  total_.fetch_add(1, std::memory_order_relaxed);
  ring_.push_back(std::move(e));
  while (ring_.size() > ring_capacity_) {
    ring_.pop_front();
  }
  if (file_) {
    file_ << line << "\n";
    file_.flush();
  }
  if (echo_.load(std::memory_order_relaxed)) {
    std::cerr << "[oracle-security] " << line << "\n";
  }
}

std::vector<SecurityEvent> AuditLog::recent(size_t n) const {
  std::lock_guard<std::mutex> g(mu_);
  const size_t take = std::min(n, ring_.size());
  return std::vector<SecurityEvent>(ring_.end() - static_cast<long>(take), ring_.end());
}

std::unordered_map<std::string, uint64_t> AuditLog::counters() const {
  std::lock_guard<std::mutex> g(mu_);
  return counters_;
}

uint64_t AuditLog::count(const std::string& category) const {
  std::lock_guard<std::mutex> g(mu_);
  const auto it = counters_.find(category);
  return it == counters_.end() ? 0 : it->second;
}

}  // namespace oracle::security

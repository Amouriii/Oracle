#include "oracle/orch/scheduler.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace oracle {
namespace {

std::string json_escape(std::string_view s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
    }
    o += (static_cast<unsigned char>(c) < 0x20) ? ' ' : c;
  }
  return o;
}

std::string make_request_id(uint64_t n) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "req-%012llx", static_cast<unsigned long long>(n));
  return buf;
}

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  if (a.time_since_epoch().count() == 0 || b.time_since_epoch().count() == 0) {
    return 0.0;
  }
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

const char* request_state_name(RequestState s) {
  switch (s) {
    case RequestState::Queued: return "queued";
    case RequestState::Running: return "running";
    case RequestState::Completed: return "completed";
    case RequestState::Failed: return "failed";
    case RequestState::TimedOut: return "timed_out";
    case RequestState::Rejected: return "rejected";
  }
  return "unknown";
}

double RequestTicket::queued_ms() const {
  const auto end = started_at.time_since_epoch().count() ? started_at : finished_at;
  return ms_between(enqueued_at, end);
}

double RequestTicket::run_ms() const { return ms_between(started_at, finished_at); }

std::string RequestTicket::to_json() const {
  std::ostringstream os;
  os << "{\"id\":\"" << json_escape(id) << "\",\"seq_id\":" << seq_id << ",\"state\":\""
     << request_state_name(state) << "\",\"priority\":" << priority << ",\"api_key\":\""
     << json_escape(api_key_id) << "\",\"model\":\"" << json_escape(model)
     << "\",\"prompt_tokens\":" << prompt_tokens << ",\"max_tokens\":" << max_tokens
     << ",\"generated_tokens\":" << generated_tokens << ",\"queued_ms\":" << queued_ms()
     << ",\"run_ms\":" << run_ms();
  if (!error.empty()) {
    os << ",\"error\":\"" << json_escape(error) << "\"";
  }
  os << "}";
  return os.str();
}

Admission& Admission::operator=(Admission&& o) noexcept {
  if (this != &o) {
    release();
    sched_ = o.sched_;
    ticket_ = std::move(o.ticket_);
    o.sched_ = nullptr;
  }
  return *this;
}

Admission::~Admission() { release(); }

void Admission::complete(uint32_t generated_tokens) {
  if (!sched_) {
    return;
  }
  ticket_.generated_tokens = generated_tokens;
  ticket_.state = RequestState::Completed;
  release();
}

void Admission::fail(const std::string& reason) {
  if (!sched_) {
    return;
  }
  ticket_.state = RequestState::Failed;
  ticket_.error = reason;
  release();
}

void Admission::release() {
  if (!sched_) {
    return;
  }
  Scheduler* s = sched_;
  sched_ = nullptr;
  if (ticket_.state == RequestState::Running) {
    // Destroyed without an explicit outcome: treat as a failure so the counters
    // stay honest rather than quietly losing the request.
    ticket_.state = RequestState::Failed;
    if (ticket_.error.empty()) {
      ticket_.error = "request ended without a result";
    }
  }
  ticket_.finished_at = std::chrono::steady_clock::now();
  s->finish(ticket_);
}

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::configure(const SchedulerConfig& cfg) {
  std::lock_guard<std::mutex> g(mu_);
  cfg_ = cfg;
  if (cfg_.max_concurrent == 0) {
    cfg_.max_concurrent = 1;
  }
  cv_.notify_all();
}

void Scheduler::set_required_stages(std::vector<LayerRange> stages) {
  std::lock_guard<std::mutex> g(mu_);
  stages_ = std::move(stages);
}

void Scheduler::set_requirement(const ResourceRequirement& need) {
  std::lock_guard<std::mutex> g(mu_);
  need_ = need;
}

Status Scheduler::stages_ready() const {
  if (!registry_ || stages_.empty()) {
    return Status::OK();
  }
  for (const auto& stage : stages_) {
    const auto best = registry_->best_for_stage(stage, need_);
    if (!best) {
      return Status::fail(Errc::worker_dead,
                          "no healthy worker owns layers [" + std::to_string(stage.start) + ", " +
                              std::to_string(stage.end) + ")");
    }
  }
  return Status::OK();
}

bool Scheduler::my_turn_locked(const std::string& id) const {
  return !queue_.empty() && queue_.front().id == id;
}

void Scheduler::record_locked(const RequestTicket& t) {
  history_.push_back(t);
  while (history_.size() > history_capacity_) {
    history_.pop_front();
  }
}

Admission Scheduler::admit(const std::string& api_key_id, const std::string& model,
                           uint32_t prompt_tokens, uint32_t max_tokens, int priority,
                           Status* why_not) {
  Admission out;
  RequestTicket t;
  t.id = make_request_id(submitted_.fetch_add(1) + 1);
  t.seq_id = next_seq_.fetch_add(1);
  t.priority = priority;
  t.api_key_id = api_key_id;
  t.model = model;
  t.prompt_tokens = prompt_tokens;
  t.max_tokens = max_tokens;
  t.enqueued_at = std::chrono::steady_clock::now();

  std::unique_lock<std::mutex> lk(mu_);
  if (stopping_) {
    t.state = RequestState::Rejected;
    t.error = "Oracle is shutting down";
    rejected_.fetch_add(1);
    record_locked(t);
    if (why_not) {
      *why_not = Status::fail(Errc::busy, t.error);
    }
    out.ticket_ = t;
    return out;
  }
  if (cfg_.max_queue_depth && queue_.size() >= cfg_.max_queue_depth) {
    t.state = RequestState::Rejected;
    t.error = "queue is full (" + std::to_string(queue_.size()) + " waiting)";
    rejected_.fetch_add(1);
    record_locked(t);
    if (why_not) {
      *why_not = Status::fail(Errc::busy, t.error);
    }
    out.ticket_ = t;
    return out;
  }

  // Insert by priority, FIFO within a band.
  Waiter w{t.id, priority, ordinal_++};
  const auto at = std::find_if(queue_.begin(), queue_.end(),
                               [&](const Waiter& q) { return q.priority < priority; });
  queue_.insert(at, w);

  const auto deadline = t.enqueued_at + std::chrono::milliseconds(
                                            cfg_.queue_timeout_ms ? cfg_.queue_timeout_ms : 30000);
  Status stage_status = Status::OK();
  const bool ready = cv_.wait_until(lk, deadline, [&] {
    if (stopping_) {
      return true;
    }
    if (!my_turn_locked(t.id) || active_ >= cfg_.max_concurrent) {
      return false;
    }
    // Only check the mesh once this request is actually next in line, so a
    // transient outage does not spin every waiter.
    stage_status = stages_ready();
    return true;
  });

  const auto drop = [&](RequestState state, const std::string& err, Errc code) {
    const auto it = std::find_if(queue_.begin(), queue_.end(),
                                 [&](const Waiter& q) { return q.id == t.id; });
    if (it != queue_.end()) {
      queue_.erase(it);
    }
    t.state = state;
    t.error = err;
    t.finished_at = std::chrono::steady_clock::now();
    record_locked(t);
    if (why_not) {
      *why_not = Status::fail(code, err);
    }
    out.ticket_ = t;
    cv_.notify_all();
  };

  if (stopping_) {
    drop(RequestState::Rejected, "Oracle is shutting down", Errc::busy);
    rejected_.fetch_add(1);
    return out;
  }
  if (!ready) {
    drop(RequestState::TimedOut,
         "waited " + std::to_string(cfg_.queue_timeout_ms) + " ms for an execution slot",
         Errc::timeout);
    timed_out_.fetch_add(1);
    return out;
  }
  if (!stage_status) {
    drop(RequestState::Rejected, stage_status.message, stage_status.code);
    rejected_.fetch_add(1);
    return out;
  }

  queue_.pop_front();
  ++active_;
  t.state = RequestState::Running;
  t.started_at = std::chrono::steady_clock::now();
  running_[t.id] = t;
  out.sched_ = this;
  out.ticket_ = t;
  if (why_not) {
    *why_not = Status::OK();
  }
  // The next waiter may also fit; wake the queue rather than only one thread.
  cv_.notify_all();
  return out;
}

void Scheduler::finish(const RequestTicket& t) {
  {
    std::lock_guard<std::mutex> g(mu_);
    running_.erase(t.id);
    if (active_ > 0) {
      --active_;
    }
    total_queue_ms_ += t.queued_ms();
    total_run_ms_ += t.run_ms();
    ++finished_;
    record_locked(t);
  }
  switch (t.state) {
    case RequestState::Completed:
      completed_.fetch_add(1);
      generated_tokens_.fetch_add(t.generated_tokens);
      break;
    case RequestState::TimedOut:
      timed_out_.fetch_add(1);
      break;
    case RequestState::Rejected:
      rejected_.fetch_add(1);
      break;
    default:
      failed_.fetch_add(1);
      break;
  }
  cv_.notify_all();
}

void Scheduler::shutdown() {
  {
    std::lock_guard<std::mutex> g(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
}

SchedulerStats Scheduler::stats() const {
  SchedulerStats s;
  s.submitted = submitted_.load();
  s.completed = completed_.load();
  s.failed = failed_.load();
  s.rejected = rejected_.load();
  s.timed_out = timed_out_.load();
  s.generated_tokens = generated_tokens_.load();
  std::lock_guard<std::mutex> g(mu_);
  s.running = active_;
  s.queued = static_cast<uint32_t>(queue_.size());
  if (finished_) {
    s.avg_queue_ms = total_queue_ms_ / static_cast<double>(finished_);
    s.avg_run_ms = total_run_ms_ / static_cast<double>(finished_);
  }
  if (total_run_ms_ > 0) {
    s.tokens_per_second = static_cast<double>(s.generated_tokens) / (total_run_ms_ / 1000.0);
  }
  return s;
}

std::vector<RequestTicket> Scheduler::recent(size_t n) const {
  std::lock_guard<std::mutex> g(mu_);
  const size_t take = std::min(n, history_.size());
  return std::vector<RequestTicket>(history_.end() - static_cast<long>(take), history_.end());
}

std::vector<RequestTicket> Scheduler::in_flight() const {
  std::lock_guard<std::mutex> g(mu_);
  std::vector<RequestTicket> out;
  out.reserve(running_.size());
  for (const auto& [id, t] : running_) {
    out.push_back(t);
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  return out;
}

std::string Scheduler::to_json() const {
  const auto s = stats();
  std::ostringstream os;
  os << "{\"config\":{\"max_concurrent\":" << cfg_.max_concurrent
     << ",\"max_queue_depth\":" << cfg_.max_queue_depth
     << ",\"queue_timeout_ms\":" << cfg_.queue_timeout_ms
     << ",\"request_timeout_ms\":" << cfg_.request_timeout_ms << "}";
  os << ",\"stats\":{\"submitted\":" << s.submitted << ",\"completed\":" << s.completed
     << ",\"failed\":" << s.failed << ",\"rejected\":" << s.rejected << ",\"timed_out\":" << s.timed_out
     << ",\"running\":" << s.running << ",\"queued\":" << s.queued
     << ",\"generated_tokens\":" << s.generated_tokens << ",\"avg_queue_ms\":" << s.avg_queue_ms
     << ",\"avg_run_ms\":" << s.avg_run_ms << ",\"tokens_per_second\":" << s.tokens_per_second << "}";
  os << ",\"active\":[";
  const auto live = in_flight();
  for (size_t i = 0; i < live.size(); ++i) {
    os << (i ? "," : "") << live[i].to_json();
  }
  os << "],\"recent\":[";
  const auto hist = recent(15);
  for (size_t i = 0; i < hist.size(); ++i) {
    os << (i ? "," : "") << hist[i].to_json();
  }
  os << "]}";
  return os.str();
}

}  // namespace oracle

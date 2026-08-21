#pragma once

// Multi-request admission and scheduling.
//
// Requests arrive on HTTP threads and are ordered in a priority queue.  A
// request runs on the thread that submitted it -- that keeps SSE streaming
// natural and gives backpressure for free -- but it only starts once the
// scheduler has confirmed a free execution slot, that its turn has come, and
// that every pipeline stage it needs is backed by a healthy worker.

#include "oracle/orch/worker_registry.hpp"
#include "oracle/types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace oracle {

enum class RequestState { Queued, Running, Completed, Failed, TimedOut, Rejected };

const char* request_state_name(RequestState s);

struct SchedulerConfig {
  uint32_t max_concurrent{4};      // requests executing at once
  uint32_t max_queue_depth{128};   // waiting requests before shedding
  uint32_t queue_timeout_ms{30000};
  uint32_t request_timeout_ms{300000};
  // Requests whose prompts are within this many tokens of each other may be
  // reported as a batch group; the pipeline still executes them separately.
  uint32_t batch_window_ms{0};
  uint32_t max_batch{1};
};

struct RequestTicket {
  std::string id;         // "req-<hex>"
  uint64_t seq_id{0};     // sequence id used on the wire and for the KV cache
  int priority{0};        // higher runs first
  std::string api_key_id;
  std::string model;
  uint32_t prompt_tokens{0};
  uint32_t max_tokens{0};
  RequestState state{RequestState::Queued};
  std::string error;
  std::chrono::steady_clock::time_point enqueued_at{};
  std::chrono::steady_clock::time_point started_at{};
  std::chrono::steady_clock::time_point finished_at{};
  uint32_t generated_tokens{0};

  [[nodiscard]] double queued_ms() const;
  [[nodiscard]] double run_ms() const;
  [[nodiscard]] std::string to_json() const;
};

struct SchedulerStats {
  uint64_t submitted{0};
  uint64_t completed{0};
  uint64_t failed{0};
  uint64_t rejected{0};
  uint64_t timed_out{0};
  uint32_t running{0};
  uint32_t queued{0};
  uint64_t generated_tokens{0};
  double avg_queue_ms{0};
  double avg_run_ms{0};
  double tokens_per_second{0};
};

// Reserves an execution slot for the lifetime of one request.  Destroying it
// releases the slot and wakes the next waiter.
class Scheduler;

class Admission {
 public:
  Admission() = default;
  Admission(const Admission&) = delete;
  Admission& operator=(const Admission&) = delete;
  Admission(Admission&& o) noexcept { *this = std::move(o); }
  Admission& operator=(Admission&& o) noexcept;
  ~Admission();

  [[nodiscard]] bool admitted() const noexcept { return sched_ != nullptr; }
  [[nodiscard]] const RequestTicket& ticket() const noexcept { return ticket_; }
  [[nodiscard]] RequestTicket& ticket() noexcept { return ticket_; }
  void complete(uint32_t generated_tokens);
  void fail(const std::string& reason);
  void release();

 private:
  friend class Scheduler;
  Scheduler* sched_{nullptr};
  RequestTicket ticket_{};
};

class Scheduler {
 public:
  Scheduler() = default;
  ~Scheduler();

  void configure(const SchedulerConfig& cfg);
  [[nodiscard]] const SchedulerConfig& config() const noexcept { return cfg_; }
  void attach_registry(WorkerRegistry* reg) { registry_ = reg; }
  // Stages the pipeline needs, in execution order.  Used to refuse a request
  // when a stage has no healthy worker instead of hanging on a dead node.
  void set_required_stages(std::vector<LayerRange> stages);
  void set_requirement(const ResourceRequirement& need);

  // Blocks until the request may run, the queue timeout expires, or the queue
  // is full.  Check `admitted()` on the result.
  [[nodiscard]] Admission admit(const std::string& api_key_id, const std::string& model,
                                uint32_t prompt_tokens, uint32_t max_tokens, int priority,
                                Status* why_not = nullptr);
  void shutdown();

  [[nodiscard]] SchedulerStats stats() const;
  [[nodiscard]] std::vector<RequestTicket> recent(size_t n = 25) const;
  [[nodiscard]] std::vector<RequestTicket> in_flight() const;
  [[nodiscard]] std::string to_json() const;
  [[nodiscard]] uint64_t next_seq_id() { return next_seq_.fetch_add(1); }

 private:
  friend class Admission;

  struct Waiter {
    std::string id;
    int priority{0};
    uint64_t ordinal{0};  // FIFO within a priority band
  };

  void release_slot(const RequestTicket& t);
  void finish(const RequestTicket& t);
  [[nodiscard]] Status stages_ready() const;
  [[nodiscard]] bool my_turn_locked(const std::string& id) const;
  void record_locked(const RequestTicket& t);

  mutable std::mutex mu_;
  std::condition_variable cv_;
  SchedulerConfig cfg_{};
  WorkerRegistry* registry_{nullptr};
  std::vector<LayerRange> stages_;
  ResourceRequirement need_{};

  std::deque<Waiter> queue_;
  std::unordered_map<std::string, RequestTicket> running_;
  std::deque<RequestTicket> history_;
  size_t history_capacity_{100};

  uint32_t active_{0};
  uint64_t ordinal_{0};
  std::atomic<uint64_t> next_seq_{1};
  std::atomic<uint64_t> submitted_{0};
  std::atomic<uint64_t> completed_{0};
  std::atomic<uint64_t> failed_{0};
  std::atomic<uint64_t> rejected_{0};
  std::atomic<uint64_t> timed_out_{0};
  std::atomic<uint64_t> generated_tokens_{0};
  double total_queue_ms_{0};
  double total_run_ms_{0};
  uint64_t finished_{0};
  bool stopping_{false};
};

}  // namespace oracle

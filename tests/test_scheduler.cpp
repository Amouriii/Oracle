// Covers admission ordering, concurrency limits, queue shedding, queue
// timeouts, dead-stage refusal and the resource-aware worker scoring.
#include "oracle/orch/scheduler.hpp"
#include "oracle/orch/worker_registry.hpp"

#include "check.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace oracle;

namespace {

WorkerResources make_worker(NodeId id, uint32_t l0, uint32_t l1) {
  WorkerResources w;
  w.id = id;
  w.host = "10.0.0." + std::to_string(id + 1);
  w.layers = {l0, l1};
  w.state = WorkerState::Ready;
  w.cpu_cores = 8;
  w.cpu_load = 0.1;
  w.ram_total_bytes = 32ull << 30;
  w.ram_free_bytes = 24ull << 30;
  w.max_concurrent = 4;
  w.last_seen = std::chrono::steady_clock::now();
  return w;
}

void check_registry() {
  WorkerRegistry reg;
  RegistryConfig cfg;
  cfg.heartbeat_interval_ms = 50;
  cfg.heartbeat_misses = 2;
  cfg.reconnect_after_ms = 10;
  reg.configure(cfg);

  reg.upsert(make_worker(0, 0, 8));
  reg.upsert(make_worker(1, 8, 16));
  CHECK(reg.size() == 2);
  CHECK(reg.alive_count() == 2);

  ResourceRequirement need;
  need.ram_bytes = 1ull << 30;
  const auto best = reg.best_for_stage({8, 16}, need);
  CHECK(best && best->id == 1);
  CHECK(!reg.best_for_stage({16, 24}, need).has_value());

  // A worker with no headroom for the request scores negative.
  auto cramped = make_worker(2, 16, 24);
  cramped.ram_free_bytes = 128ull << 20;
  reg.upsert(cramped);
  CHECK(reg.score(cramped, need) < 0);
  CHECK(!reg.best_for_stage({16, 24}, need).has_value());

  // Between two viable candidates, the less loaded one wins.
  auto busy = make_worker(3, 24, 32);
  busy.active_requests = 3;
  busy.cpu_load = 0.9;
  auto idle = make_worker(4, 24, 32);
  idle.link_latency_ms = 0.2;
  reg.upsert(busy);
  reg.upsert(idle);
  CHECK(reg.score(idle, need) > reg.score(busy, need));
  const auto chosen = reg.best_for_stage({24, 32}, need);
  CHECK(chosen && chosen->id == 4);

  // A worker at its concurrency cap is not accepting.
  auto full = idle;
  full.active_requests = full.max_concurrent;
  CHECK(!full.accepting());
  CHECK(reg.score(full, need) < 0);

  // Heartbeat payloads round-trip through the datagram encoding.
  auto reporter = make_worker(5, 32, 40);
  reporter.cpu_load = 0.42;
  reporter.ram_free_bytes = 7ull << 30;
  reporter.active_requests = 2;
  reporter.runner = "gguf";
  reporter.model = "tiny";
  reg.note_heartbeat(5, reporter.encode_heartbeat());
  const auto got = reg.get(5);
  CHECK(got);
  CHECK(got->cpu_load > 0.41 && got->cpu_load < 0.43);
  CHECK(got->ram_free_bytes == (7ull << 30));
  CHECK(got->active_requests == 2);
  CHECK(got->runner == "gguf");
  CHECK(got->layers.start == 32 && got->layers.end == 40);
  CHECK(got->state == WorkerState::Busy);

  // Missing heartbeats eventually mark the node dead.
  std::this_thread::sleep_for(std::chrono::milliseconds(160));
  std::vector<NodeId> dead;
  for (int i = 0; i < 4 && dead.empty(); ++i) {
    dead = reg.tick();
    if (dead.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(60));
    }
  }
  CHECK(!dead.empty());
  const auto after = reg.get(dead.front());
  CHECK(after && after->state == WorkerState::Dead);
  CHECK(!after->last_error.empty());
  CHECK(!after->accepting());

  // A dead node becomes a reconnect candidate once the backoff has elapsed.
  CHECK(reg.due_for_reconnect().empty());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  CHECK(!reg.due_for_reconnect().empty());

  // A heartbeat from a dead node brings it straight back.
  reg.note_heartbeat(dead.front(), "");
  CHECK(reg.healthy(dead.front()));
  CHECK(reg.to_json().find("\"id\":0") != std::string::npos);

  // A node that misses a beat and then comes back must return to Ready rather
  // than staying Degraded forever.
  reg.upsert(make_worker(7, 48, 56));
  reg.mark_state(7, WorkerState::Degraded);
  CHECK(!reg.healthy(7));
  reg.note_heartbeat(7, "");
  CHECK(reg.healthy(7));
  CHECK(reg.get(7)->state == WorkerState::Ready);

  // Heartbeats travel over UDP and activations over TCP.  A node answering
  // heartbeats while its activation link is down must NOT count as healthy --
  // otherwise a restarted worker looks alive and requests are dispatched into a
  // connection that no longer exists.
  reg.upsert(make_worker(6, 40, 48));
  CHECK(reg.healthy(6));
  reg.set_link_up(6, false);
  reg.note_heartbeat(6, "");
  CHECK(!reg.healthy(6));
  CHECK(!reg.best_for_stage({40, 48}, need).has_value());
  reg.set_link_up(6, true);
  CHECK(reg.healthy(6));
  CHECK(reg.best_for_stage({40, 48}, need).has_value());
}

void check_admission_order_and_limits() {
  Scheduler s;
  SchedulerConfig cfg;
  cfg.max_concurrent = 1;
  cfg.max_queue_depth = 2;
  cfg.queue_timeout_ms = 200;
  s.configure(cfg);

  Status why;
  auto first = s.admit("k", "m", 10, 16, 0, &why);
  CHECK(first.admitted());
  CHECK(why.ok());
  CHECK(first.ticket().state == RequestState::Running);
  CHECK(!first.ticket().id.empty());
  CHECK(first.ticket().seq_id > 0);

  // A second request cannot start while the slot is taken, so it times out.
  const auto t0 = std::chrono::steady_clock::now();
  auto second = s.admit("k", "m", 10, 16, 0, &why);
  const auto waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                          .count();
  CHECK(!second.admitted());
  CHECK(why.code == Errc::timeout);
  CHECK(waited >= 150.0);
  CHECK(second.ticket().state == RequestState::TimedOut);

  first.complete(12);
  const auto st = s.stats();
  CHECK(st.completed == 1);
  CHECK(st.generated_tokens == 12);
  CHECK(st.timed_out == 1);
  CHECK(st.running == 0);

  // Once the slot frees up the next request runs immediately.
  auto third = s.admit("k", "m", 10, 16, 0, &why);
  CHECK(third.admitted());
  third.fail("simulated failure");
  CHECK(s.stats().failed == 1);

  // Dropping an admission without an outcome must still free the slot.
  {
    auto scoped = s.admit("k", "m", 1, 1, 0, &why);
    CHECK(scoped.admitted());
  }
  CHECK(s.stats().running == 0);
  auto after_scope = s.admit("k", "m", 1, 1, 0, &why);
  CHECK(after_scope.admitted());
  after_scope.complete(1);
}

void check_priority() {
  Scheduler s;
  SchedulerConfig cfg;
  cfg.max_concurrent = 1;
  cfg.max_queue_depth = 16;
  cfg.queue_timeout_ms = 5000;
  s.configure(cfg);

  auto blocker = s.admit("k", "m", 1, 1, 0, nullptr);
  CHECK(blocker.admitted());

  // Enqueued as normal-a, high, normal-b.  Expected service order is the high
  // priority first, then the two normal ones in the order they arrived.
  struct Job {
    const char* label;
    int priority;
  };
  const Job jobs[] = {{"normal-a", 0}, {"high", 5}, {"normal-b", 0}};

  std::vector<std::string> order;
  std::mutex om;
  std::vector<std::thread> threads;
  for (const auto& job : jobs) {
    threads.emplace_back([&, job] {
      auto a = s.admit("k", "m", 1, 1, job.priority, nullptr);
      if (a.admitted()) {
        {
          std::lock_guard<std::mutex> g(om);
          order.emplace_back(job.label);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        a.complete(1);
      }
    });
    // Stagger so the queue order is deterministic before the slot frees.
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  blocker.complete(1);
  for (auto& t : threads) {
    t.join();
  }
  CHECK(order.size() == 3);
  CHECK(order[0] == "high");
  CHECK(order[1] == "normal-a");
  CHECK(order[2] == "normal-b");
}

void check_queue_shedding() {
  Scheduler s;
  SchedulerConfig cfg;
  cfg.max_concurrent = 1;
  cfg.max_queue_depth = 1;
  cfg.queue_timeout_ms = 400;
  s.configure(cfg);

  auto running = s.admit("k", "m", 1, 1, 0, nullptr);
  CHECK(running.admitted());

  // One waiter fills the queue; the next is shed immediately.
  std::thread waiter([&] {
    auto a = s.admit("k", "m", 1, 1, 0, nullptr);
    (void)a.admitted();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  Status why;
  auto shed = s.admit("k", "m", 1, 1, 0, &why);
  CHECK(!shed.admitted());
  CHECK(why.code == Errc::busy);
  CHECK(shed.ticket().state == RequestState::Rejected);
  running.complete(1);
  waiter.join();
  CHECK(s.stats().rejected >= 1);
}

void check_dead_stage_refusal() {
  WorkerRegistry reg;
  RegistryConfig rc;
  rc.heartbeat_interval_ms = 50;
  rc.heartbeat_misses = 2;
  reg.configure(rc);
  reg.upsert(make_worker(0, 0, 8));
  reg.upsert(make_worker(1, 8, 16));

  Scheduler s;
  SchedulerConfig cfg;
  cfg.max_concurrent = 2;
  cfg.queue_timeout_ms = 300;
  s.configure(cfg);
  s.attach_registry(&reg);
  s.set_required_stages({{0, 8}, {8, 16}});

  Status why;
  auto ok = s.admit("k", "m", 1, 1, 0, &why);
  CHECK(ok.admitted());
  ok.complete(1);

  // With a stage's only worker dead, admission is refused with a clear reason
  // rather than dispatching into a black hole.
  reg.mark_dead(1, "link down");
  auto refused = s.admit("k", "m", 1, 1, 0, &why);
  CHECK(!refused.admitted());
  CHECK(why.code == Errc::worker_dead);
  CHECK(why.message.find("[8, 16)") != std::string::npos);

  // Recovery re-opens admission.
  reg.note_reconnect(1);
  auto back = s.admit("k", "m", 1, 1, 0, &why);
  CHECK(back.admitted());
  back.complete(1);

  const auto json = s.to_json();
  CHECK(json.find("\"max_concurrent\":2") != std::string::npos);
  CHECK(json.find("\"recent\"") != std::string::npos);
}

}  // namespace

int main() {
  check_registry();
  check_admission_order_and_limits();
  check_priority();
  check_queue_shedding();
  check_dead_stage_refusal();
  std::cout << "test_scheduler ok\n";
  return 0;
}

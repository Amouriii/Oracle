#include "oracle/shard/memory_shard_manager.hpp"

#if defined(__APPLE__)
#include <mach/host_info.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#else
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#endif

namespace oracle {
namespace {

#if !defined(__APPLE__)
// cgroup v2/v1 limits are what actually bounds a container, and /proc/meminfo
// reports the host's totals, so a worker in Docker must consult both.
uint64_t read_first_u64(const char* path, uint64_t fallback) {
  std::ifstream in(path);
  if (!in) {
    return fallback;
  }
  std::string tok;
  if (!(in >> tok)) {
    return fallback;
  }
  if (tok == "max") {
    return fallback;
  }
  try {
    return std::stoull(tok);
  } catch (...) {
    return fallback;
  }
}
#endif

}  // namespace

MemorySnapshot PressureMonitor::sample() const {
  MemorySnapshot s;
#if defined(__APPLE__)
  int64_t memsize = 0;
  size_t len = sizeof(memsize);
  sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
  s.total_bytes = static_cast<uint64_t>(memsize);

  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_statistics64_data_t vmstat{};
  host_t host = mach_host_self();
  if (host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vmstat), &count) == KERN_SUCCESS) {
    const uint64_t page = static_cast<uint64_t>(vm_kernel_page_size);
    const uint64_t free_p = static_cast<uint64_t>(vmstat.free_count) + static_cast<uint64_t>(vmstat.inactive_count) +
                            static_cast<uint64_t>(vmstat.purgeable_count);
    s.free_bytes = free_p * page;
    s.compressed_bytes = static_cast<uint64_t>(vmstat.compressor_page_count) * page;
    s.used_bytes = s.total_bytes > s.free_bytes ? s.total_bytes - s.free_bytes : 0;
    s.under_pressure = vmstat.free_count < (vmstat.speculative_count + 1024);
  }
  // VRAM budget is filled by MetalNodeRunner via recommendedMaxWorkingSetSize when available.
#else
  std::ifstream in("/proc/meminfo");
  uint64_t mem_total_kb = 0, mem_available_kb = 0, swap_total_kb = 0, swap_free_kb = 0;
  std::string key;
  uint64_t value = 0;
  std::string unit;
  while (in >> key >> value) {
    std::getline(in, unit);
    if (key == "MemTotal:") {
      mem_total_kb = value;
    } else if (key == "MemAvailable:") {
      mem_available_kb = value;
    } else if (key == "SwapTotal:") {
      swap_total_kb = value;
    } else if (key == "SwapFree:") {
      swap_free_kb = value;
    }
  }
  s.total_bytes = mem_total_kb * 1024ull;
  s.free_bytes = mem_available_kb * 1024ull;
  if (s.total_bytes == 0) {
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0) {
      s.total_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page);
      s.free_bytes = s.total_bytes / 2;
    }
  }

  // Containers: honour the cgroup limit when it is tighter than the host total.
  const uint64_t cg_max = read_first_u64("/sys/fs/cgroup/memory.max", 0);
  const uint64_t cg_max_v1 = read_first_u64("/sys/fs/cgroup/memory/memory.limit_in_bytes", 0);
  uint64_t limit = cg_max ? cg_max : cg_max_v1;
  if (limit && limit < s.total_bytes) {
    const uint64_t cur = read_first_u64("/sys/fs/cgroup/memory.current",
                                        read_first_u64("/sys/fs/cgroup/memory/memory.usage_in_bytes", 0));
    s.total_bytes = limit;
    s.free_bytes = limit > cur ? limit - cur : 0;
  }

  s.used_bytes = s.total_bytes > s.free_bytes ? s.total_bytes - s.free_bytes : 0;
  s.compressed_bytes = (swap_total_kb > swap_free_kb) ? (swap_total_kb - swap_free_kb) * 1024ull : 0;
  s.under_pressure = s.total_bytes > 0 && s.free_bytes * 10 < s.total_bytes;  // < 10% available
#endif
  return s;
}

Status PressureMonitor::refuse_if_overcommit(uint64_t extra_bytes, uint64_t ram_budget_bytes) const {
  const MemorySnapshot snap = sample();
  const uint64_t cap = ram_budget_bytes ? ram_budget_bytes : snap.total_bytes;
  const uint64_t need = extra_bytes;
  if (need > cap) {
    return Status::fail(Errc::pressure, "allocation " + std::to_string(need) + " exceeds budget " + std::to_string(cap));
  }
  if (snap.free_bytes > 0 && need > snap.free_bytes + (snap.free_bytes / 4)) {
    return Status::fail(Errc::pressure, "host would overcommit: need " + std::to_string(need) + " free " +
                                            std::to_string(snap.free_bytes));
  }
  if (snap.under_pressure && need > (64ull << 20)) {
    return Status::fail(Errc::pressure, "memory_pressure is high");
  }
  return Status::OK();
}

}  // namespace oracle

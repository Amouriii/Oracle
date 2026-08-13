#include "oracle/shard/memory_shard_manager.hpp"

#if defined(__APPLE__)
#include <mach/host_info.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#endif


namespace oracle {

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
  s.total_bytes = 1ull << 32;
  s.free_bytes = 1ull << 31;
  s.used_bytes = s.total_bytes - s.free_bytes;
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

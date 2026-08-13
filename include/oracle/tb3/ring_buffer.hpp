#pragma once

#include "oracle/types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace oracle {

struct RingControl {
  uint32_t magic;
  uint32_t slots;  // power of two
  uint64_t payload_bytes;
  uint64_t slot_stride;
  alignas(64) std::atomic<uint64_t> head;
  alignas(64) std::atomic<uint64_t> tail;
};

class LockFreeRingBuffer {
 public:
  LockFreeRingBuffer() = default;
  ~LockFreeRingBuffer();
  LockFreeRingBuffer(const LockFreeRingBuffer&) = delete;
  LockFreeRingBuffer& operator=(const LockFreeRingBuffer&) = delete;
  LockFreeRingBuffer(LockFreeRingBuffer&&) noexcept;
  LockFreeRingBuffer& operator=(LockFreeRingBuffer&&) noexcept;

  static Status create(std::string_view shm_name, uint32_t slots, uint64_t payload_bytes,
                       LockFreeRingBuffer* out);
  static Status open(std::string_view shm_name, LockFreeRingBuffer* out);
  static Status create_anonymous(uint32_t slots, uint64_t payload_bytes, LockFreeRingBuffer* out);

  [[nodiscard]] uint32_t slots() const noexcept;
  [[nodiscard]] uint64_t payload_capacity() const noexcept;
  [[nodiscard]] uint64_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] bool full() const noexcept;
  [[nodiscard]] const std::string& name() const noexcept { return name_; }

  bool try_push(const TensorHeader& hdr, std::span<const std::byte> payload);
  bool try_pop(TensorHeader* hdr, std::span<std::byte> out, uint64_t* nbytes);
  bool try_pop_into(Tensor* out);

  Status unlink();

 private:
  void reset() noexcept;
  [[nodiscard]] uint8_t* slot_ptr(uint64_t idx) const noexcept;

  std::string name_;
  int fd_{-1};
  void* map_{nullptr};
  size_t map_bytes_{0};
  RingControl* ctl_{nullptr};
  uint8_t* slots_base_{nullptr};
  bool owner_{false};
};

}  // namespace oracle

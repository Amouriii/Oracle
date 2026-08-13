#include "oracle/tb3/ring_buffer.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace oracle {
namespace {

constexpr uint32_t kRingMagic = 0x52424F52u;  // 'ROBR'
constexpr size_t kControlBytes = 4096;

bool is_pow2(uint32_t x) { return x >= 2 && (x & (x - 1)) == 0; }

size_t map_size(uint32_t slots, uint64_t payload, uint64_t* stride) {
  const uint64_t header = sizeof(TensorHeader);
  const uint64_t align = 64;
  uint64_t s = header + payload;
  s = (s + align - 1) & ~(align - 1);
  *stride = s;
  return static_cast<size_t>(kControlBytes + slots * s);
}

std::string shm_path(std::string_view name) {
  if (!name.empty() && name.front() == '/') {
    return std::string(name);
  }
  return "/" + std::string(name);
}

}  // namespace

LockFreeRingBuffer::~LockFreeRingBuffer() { reset(); }

LockFreeRingBuffer::LockFreeRingBuffer(LockFreeRingBuffer&& o) noexcept { *this = std::move(o); }

LockFreeRingBuffer& LockFreeRingBuffer::operator=(LockFreeRingBuffer&& o) noexcept {
  if (this == &o) {
    return *this;
  }
  reset();
  name_ = std::move(o.name_);
  fd_ = o.fd_;
  map_ = o.map_;
  map_bytes_ = o.map_bytes_;
  ctl_ = o.ctl_;
  slots_base_ = o.slots_base_;
  owner_ = o.owner_;
  o.fd_ = -1;
  o.map_ = nullptr;
  o.map_bytes_ = 0;
  o.ctl_ = nullptr;
  o.slots_base_ = nullptr;
  o.owner_ = false;
  return *this;
}

void LockFreeRingBuffer::reset() noexcept {
  if (map_ && map_bytes_) {
    munmap(map_, map_bytes_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
  map_ = nullptr;
  fd_ = -1;
  map_bytes_ = 0;
  ctl_ = nullptr;
  slots_base_ = nullptr;
  owner_ = false;
}

uint8_t* LockFreeRingBuffer::slot_ptr(uint64_t idx) const noexcept {
  const uint32_t mask = ctl_->slots - 1;
  return slots_base_ + (idx & mask) * ctl_->slot_stride;
}

uint32_t LockFreeRingBuffer::slots() const noexcept { return ctl_ ? ctl_->slots : 0; }
uint64_t LockFreeRingBuffer::payload_capacity() const noexcept { return ctl_ ? ctl_->payload_bytes : 0; }

uint64_t LockFreeRingBuffer::size() const noexcept {
  if (!ctl_) {
    return 0;
  }
  const uint64_t h = ctl_->head.load(std::memory_order_acquire);
  const uint64_t t = ctl_->tail.load(std::memory_order_acquire);
  return h - t;
}

bool LockFreeRingBuffer::empty() const noexcept { return size() == 0; }
bool LockFreeRingBuffer::full() const noexcept { return ctl_ && size() >= ctl_->slots; }

Status LockFreeRingBuffer::create_anonymous(uint32_t slots, uint64_t payload_bytes, LockFreeRingBuffer* out) {
  if (!out || !is_pow2(slots) || payload_bytes == 0) {
    return Status::fail(Errc::invalid_argument, "ring: slots must be power of two, payload > 0");
  }
  uint64_t stride = 0;
  const size_t bytes = map_size(slots, payload_bytes, &stride);
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return Status::fail(Errc::oom, std::string("mmap: ") + std::strerror(errno));
  }
  out->reset();
  out->map_ = p;
  out->map_bytes_ = bytes;
  out->fd_ = -1;
  out->owner_ = true;
  out->ctl_ = static_cast<RingControl*>(p);
  new (out->ctl_) RingControl{};
  out->ctl_->magic = kRingMagic;
  out->ctl_->slots = slots;
  out->ctl_->payload_bytes = payload_bytes;
  out->ctl_->slot_stride = stride;
  out->ctl_->head.store(0, std::memory_order_relaxed);
  out->ctl_->tail.store(0, std::memory_order_relaxed);
  out->slots_base_ = static_cast<uint8_t*>(p) + kControlBytes;
  return Status::OK();
}

Status LockFreeRingBuffer::create(std::string_view shm_name, uint32_t slots, uint64_t payload_bytes,
                                  LockFreeRingBuffer* out) {
  if (!out || shm_name.empty() || !is_pow2(slots) || payload_bytes == 0) {
    return Status::fail(Errc::invalid_argument, "ring create: bad args");
  }
  uint64_t stride = 0;
  const size_t bytes = map_size(slots, payload_bytes, &stride);
  const std::string path = shm_path(shm_name);
  shm_unlink(path.c_str());
  int fd = shm_open(path.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
  if (fd < 0) {
    return Status::fail(Errc::io, std::string("shm_open create: ") + std::strerror(errno));
  }
  if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    const int e = errno;
    close(fd);
    shm_unlink(path.c_str());
    return Status::fail(Errc::io, std::string("ftruncate: ") + std::strerror(e));
  }
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    const int e = errno;
    close(fd);
    shm_unlink(path.c_str());
    return Status::fail(Errc::oom, std::string("mmap shm: ") + std::strerror(e));
  }
  out->reset();
  out->name_ = path;
  out->fd_ = fd;
  out->map_ = p;
  out->map_bytes_ = bytes;
  out->owner_ = true;
  out->ctl_ = static_cast<RingControl*>(p);
  new (out->ctl_) RingControl{};
  out->ctl_->magic = kRingMagic;
  out->ctl_->slots = slots;
  out->ctl_->payload_bytes = payload_bytes;
  out->ctl_->slot_stride = stride;
  out->ctl_->head.store(0, std::memory_order_relaxed);
  out->ctl_->tail.store(0, std::memory_order_relaxed);
  out->slots_base_ = static_cast<uint8_t*>(p) + kControlBytes;
  return Status::OK();
}

Status LockFreeRingBuffer::open(std::string_view shm_name, LockFreeRingBuffer* out) {
  if (!out || shm_name.empty()) {
    return Status::fail(Errc::invalid_argument, "ring open: bad args");
  }
  const std::string path = shm_path(shm_name);
  int fd = shm_open(path.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    return Status::fail(Errc::not_found, std::string("shm_open: ") + std::strerror(errno));
  }
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    const int e = errno;
    close(fd);
    return Status::fail(Errc::io, std::strerror(e));
  }
  void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    const int e = errno;
    close(fd);
    return Status::fail(Errc::oom, std::strerror(e));
  }
  auto* ctl = static_cast<RingControl*>(p);
  if (ctl->magic != kRingMagic) {
    munmap(p, static_cast<size_t>(st.st_size));
    close(fd);
    return Status::fail(Errc::protocol, "ring magic mismatch");
  }
  out->reset();
  out->name_ = path;
  out->fd_ = fd;
  out->map_ = p;
  out->map_bytes_ = static_cast<size_t>(st.st_size);
  out->owner_ = false;
  out->ctl_ = ctl;
  out->slots_base_ = static_cast<uint8_t*>(p) + kControlBytes;
  return Status::OK();
}

Status LockFreeRingBuffer::unlink() {
  if (name_.empty()) {
    return Status::OK();
  }
  if (shm_unlink(name_.c_str()) != 0 && errno != ENOENT) {
    return Status::fail(Errc::io, std::strerror(errno));
  }
  return Status::OK();
}

bool LockFreeRingBuffer::try_push(const TensorHeader& hdr, std::span<const std::byte> payload) {
  if (!ctl_) {
    return false;
  }
  if (payload.size() > ctl_->payload_bytes) {
    return false;
  }
  const uint64_t head = ctl_->head.load(std::memory_order_relaxed);
  const uint64_t tail = ctl_->tail.load(std::memory_order_acquire);
  if (head - tail >= ctl_->slots) {
    return false;
  }
  uint8_t* slot = slot_ptr(head);
  std::memcpy(slot, &hdr, sizeof(TensorHeader));
  if (!payload.empty()) {
    std::memcpy(slot + sizeof(TensorHeader), payload.data(), payload.size());
  }
  std::atomic_thread_fence(std::memory_order_release);
  ctl_->head.store(head + 1, std::memory_order_release);
  return true;
}

bool LockFreeRingBuffer::try_pop(TensorHeader* hdr, std::span<std::byte> out, uint64_t* nbytes) {
  if (!ctl_ || !hdr) {
    return false;
  }
  const uint64_t tail = ctl_->tail.load(std::memory_order_relaxed);
  const uint64_t head = ctl_->head.load(std::memory_order_acquire);
  if (tail == head) {
    return false;
  }
  uint8_t* slot = slot_ptr(tail);
  std::memcpy(hdr, slot, sizeof(TensorHeader));
  const uint64_t n = hdr->nbytes;
  if (n > ctl_->payload_bytes) {
    return false;
  }
  if (out.size() < n) {
    return false;
  }
  if (n > 0) {
    std::memcpy(out.data(), slot + sizeof(TensorHeader), static_cast<size_t>(n));
  }
  if (nbytes) {
    *nbytes = n;
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  ctl_->tail.store(tail + 1, std::memory_order_release);
  return true;
}

bool LockFreeRingBuffer::try_pop_into(Tensor* out) {
  if (!out || !ctl_) {
    return false;
  }
  const uint64_t tail = ctl_->tail.load(std::memory_order_relaxed);
  const uint64_t head = ctl_->head.load(std::memory_order_acquire);
  if (tail == head) {
    return false;
  }
  uint8_t* slot = slot_ptr(tail);
  TensorHeader hdr{};
  std::memcpy(&hdr, slot, sizeof(TensorHeader));
  out->header = hdr;
  out->payload.resize(static_cast<size_t>(hdr.nbytes));
  if (hdr.nbytes > 0) {
    std::memcpy(out->payload.data(), slot + sizeof(TensorHeader), static_cast<size_t>(hdr.nbytes));
  }
  ctl_->tail.store(tail + 1, std::memory_order_release);
  return true;
}

}  // namespace oracle

#include "oracle/tb3/ring_buffer.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  oracle::LockFreeRingBuffer ring;
  auto st = oracle::LockFreeRingBuffer::create_anonymous(8, 256, &ring);
  if (!st) {
    std::cerr << st.message << "\n";
    return 1;
  }
  assert(ring.slots() == 8);
  assert(ring.empty());

  oracle::TensorHeader hdr;
  hdr.rank = 1;
  hdr.shape[0] = 4;
  hdr.nbytes = 8;
  hdr.seq_id = 42;
  hdr.flags = oracle::kFlagDecode;
  const uint8_t raw[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  auto payload = std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw), 8);
  hdr.checksum = oracle::crc32(payload);
  assert(ring.try_push(hdr, payload));
  assert(!ring.empty());

  oracle::TensorHeader outh{};
  std::vector<std::byte> buf(8);
  uint64_t n = 0;
  assert(ring.try_pop(&outh, buf, &n));
  assert(n == 8);
  assert(outh.seq_id == 42);
  assert(std::memcmp(buf.data(), raw, 8) == 0);
  assert(ring.empty());

  oracle::LockFreeRingBuffer named;
  st = oracle::LockFreeRingBuffer::create("oracle-test-ring", 4, 64, &named);
  if (!st) {
    std::cerr << "named create: " << st.message << "\n";
    return 1;
  }
  oracle::LockFreeRingBuffer opened;
  st = oracle::LockFreeRingBuffer::open("oracle-test-ring", &opened);
  if (!st) {
    std::cerr << "named open: " << st.message << "\n";
    return 1;
  }
  hdr.nbytes = 4;
  const uint8_t a[4] = {9, 8, 7, 6};
  assert(named.try_push(hdr, std::span<const std::byte>(reinterpret_cast<const std::byte*>(a), 4)));
  oracle::Tensor t;
  assert(opened.try_pop_into(&t));
  assert(t.payload.size() == 4);
  named.unlink();
  std::cout << "test_ring_buffer ok\n";
  return 0;
}

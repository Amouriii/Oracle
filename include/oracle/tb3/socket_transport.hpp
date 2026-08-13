#pragma once

#include "oracle/tb3/ring_buffer.hpp"
#include "oracle/types.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace oracle {

struct TransportOptions {
  std::string bind_host{"0.0.0.0"};
  uint16_t port{9200};
  uint16_t heartbeat_port{9100};
  int backlog{16};
  bool tcp_nodelay{true};
  bool reuseaddr{true};
  uint32_t mtu{9000};
  int send_timeout_ms{5000};
  int recv_timeout_ms{5000};
  uint32_t local_ring_slots{1024};
  uint64_t local_payload_bytes{8192 * 2};  // hidden_dim * f16 default
};

class TB3SocketTransport {
 public:
  using RecvHandler = std::function<void(NodeId from, Tensor&&)>;

  TB3SocketTransport() = default;
  ~TB3SocketTransport();
  TB3SocketTransport(const TB3SocketTransport&) = delete;
  TB3SocketTransport& operator=(const TB3SocketTransport&) = delete;

  Status listen(const TransportOptions& opt);
  Status connect(NodeId peer, const std::string& host, uint16_t port);
  Status accept_one(NodeId expected_peer, int timeout_ms);

  Status send_tensor(NodeId peer, const TensorHeader& hdr, std::span<const std::byte> payload);
  Status send_tensor(NodeId peer, const Tensor& t);
  Status recv_tensor(NodeId peer, Tensor* out, int timeout_ms);

  Status send_local(const TensorHeader& hdr, std::span<const std::byte> payload);
  Status recv_local(Tensor* out);

  Status start_heartbeat(NodeId self, const std::vector<std::pair<std::string, uint16_t>>& peers);
  void stop();

  void set_recv_handler(RecvHandler h) { handler_ = std::move(h); }
  [[nodiscard]] bool listening() const noexcept { return listen_fd_ >= 0; }
  [[nodiscard]] uint16_t port() const noexcept { return opt_.port; }

  LockFreeRingBuffer& local_ring() { return local_ring_; }

 private:
  struct Conn {
    NodeId peer{0};
    int fd{-1};
    std::vector<std::byte> rbuf;
  };

  Status send_fd(int fd, const TensorHeader& hdr, std::span<const std::byte> payload);
  Status recv_fd(int fd, Tensor* out, int timeout_ms);
  int find_fd(NodeId peer) const;
  void io_loop();
  void hb_loop();

  TransportOptions opt_{};
  int listen_fd_{-1};
  int kq_{-1};
  int hb_fd_{-1};
  std::atomic<bool> running_{false};
  std::thread io_thread_;
  std::thread hb_thread_;
  mutable std::mutex mu_;
  std::vector<Conn> conns_;
  RecvHandler handler_;
  LockFreeRingBuffer local_ring_;
  NodeId self_id_{0};
  std::vector<std::pair<std::string, uint16_t>> hb_peers_;
};

}  // namespace oracle

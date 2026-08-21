#pragma once

// Point-to-point tensor transport between mesh nodes.
//
// The wire format is a packed 76-byte TensorHeader followed by the raw payload,
// written with a single writev() so the header never costs an extra syscall or
// an extra segment.  Each peer connection carries its own send and receive
// locks, so a master can be streaming activations to the next hop while it is
// still reading logits back from the last one.
//
// A local hop (two Oracle processes on the same machine) can bypass the socket
// entirely through a POSIX shared-memory SPSC ring.

#include "oracle/tb3/ring_buffer.hpp"
#include "oracle/types.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
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
  uint64_t local_payload_bytes{8192 * 2};
  // Socket buffer sizes.  A 40 Gb/s Thunderbolt link needs a window far larger
  // than the default to keep a single stream saturated.
  int send_buffer_bytes{4 << 20};
  int recv_buffer_bytes{4 << 20};
  // Refuse a frame whose declared payload exceeds this, before allocating.
  uint64_t max_payload_bytes{1ull << 30};
};

// What a joining node presents to the master.
struct Handshake {
  NodeId node_id{0};
  std::string role{"worker"};
  uint32_t protocol_version{kProtocolVersion};
  std::string nonce;
  std::string signature;  // HMAC-SHA256 hex over "<node_id>:<nonce>"
  std::string runner;
  uint32_t layer_start{0};
  uint32_t layer_end{0};

  [[nodiscard]] std::string encode() const;
  static Status decode(std::string_view blob, Handshake* out);
};

struct PeerStats {
  NodeId id{0};
  std::string host;
  uint16_t port{0};
  bool connected{false};
  uint64_t bytes_sent{0};
  uint64_t bytes_recv{0};
  uint64_t frames_sent{0};
  uint64_t frames_recv{0};
  uint64_t reconnects{0};
  uint64_t errors{0};
  double last_send_ms{0};
  double last_recv_ms{0};
};

struct TransportTotals {
  uint64_t bytes_sent{0};
  uint64_t bytes_recv{0};
  uint64_t frames_sent{0};
  uint64_t frames_recv{0};
  uint64_t reconnects{0};
  uint64_t errors{0};
  uint64_t local_frames{0};
  size_t peers{0};
  size_t connected_peers{0};
};

class TB3SocketTransport {
 public:
  using RecvHandler = std::function<void(NodeId from, Tensor&&)>;
  // Returns true to admit a joining node.  Called with the peer's handshake.
  using AuthHandler = std::function<bool(const Handshake&, std::string* reason)>;

  TB3SocketTransport() = default;
  ~TB3SocketTransport();
  TB3SocketTransport(const TB3SocketTransport&) = delete;
  TB3SocketTransport& operator=(const TB3SocketTransport&) = delete;

  Status listen(const TransportOptions& opt);
  Status connect(NodeId peer, const std::string& host, uint16_t port);
  // Connects if needed, retrying with backoff until `timeout_ms` elapses.
  Status ensure_connected(NodeId peer, const std::string& host, uint16_t port, int timeout_ms = 10000);
  Status accept_one(NodeId expected_peer, int timeout_ms);
  // Accepts a connection and reads the joining node's handshake, consulting the
  // auth handler before the peer is registered.
  Status accept_registration(int timeout_ms, Handshake* out);
  // Sends our handshake on an already-connected peer and waits for the ack.
  Status register_with(NodeId peer, const Handshake& hello, int timeout_ms = 5000);

  Status send_tensor(NodeId peer, const TensorHeader& hdr, std::span<const std::byte> payload);
  Status send_tensor(NodeId peer, const Tensor& t);
  Status recv_tensor(NodeId peer, Tensor* out, int timeout_ms);
  // Waits for a frame from any connected peer.  `from` receives the sender.
  Status recv_any(NodeId* from, Tensor* out, int timeout_ms);

  Status send_local(const TensorHeader& hdr, std::span<const std::byte> payload);
  Status recv_local(Tensor* out);

  Status start_heartbeat(NodeId self, const std::vector<std::pair<std::string, uint16_t>>& peers);
  void set_heartbeat_payload(std::string payload);
  void stop();
  void disconnect(NodeId peer);

  void set_recv_handler(RecvHandler h);
  void set_auth_handler(AuthHandler h);
  void set_handshake(Handshake h);

  [[nodiscard]] bool listening() const noexcept { return listen_fd_ >= 0; }
  [[nodiscard]] bool connected(NodeId peer) const;
  [[nodiscard]] uint16_t port() const noexcept { return opt_.port; }
  [[nodiscard]] std::vector<PeerStats> peers() const;
  [[nodiscard]] TransportTotals totals() const;

  LockFreeRingBuffer& local_ring() { return local_ring_; }

 private:
  struct Conn {
    NodeId peer{0};
    int fd{-1};
    std::string host;
    uint16_t port{0};
    std::mutex send_mu;
    std::mutex recv_mu;
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_recv{0};
    std::atomic<uint64_t> frames_sent{0};
    std::atomic<uint64_t> frames_recv{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> reconnects{0};
    std::atomic<double> last_send_ms{0};
    std::atomic<double> last_recv_ms{0};
  };

  std::shared_ptr<Conn> find(NodeId peer) const;
  std::shared_ptr<Conn> adopt(NodeId peer, int fd, std::string host, uint16_t port);
  Status send_fd(Conn& c, const TensorHeader& hdr, std::span<const std::byte> payload);
  Status recv_fd(Conn& c, Tensor* out, int timeout_ms);
  void tune_socket(int fd) const;
  void hb_loop();

  TransportOptions opt_{};
  int listen_fd_{-1};
  int hb_fd_{-1};
  std::atomic<bool> running_{false};
  std::thread hb_thread_;
  mutable std::mutex mu_;
  std::unordered_map<NodeId, std::shared_ptr<Conn>> conns_;
  RecvHandler handler_;
  AuthHandler auth_;
  Handshake hello_{};
  LockFreeRingBuffer local_ring_;
  std::atomic<uint64_t> local_frames_{0};
  std::atomic<uint64_t> reconnects_{0};
  NodeId self_id_{0};
  std::vector<std::pair<std::string, uint16_t>> hb_peers_;
  mutable std::mutex hb_mu_;
  std::string hb_payload_;
};

}  // namespace oracle

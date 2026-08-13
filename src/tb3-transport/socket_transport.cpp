#include "oracle/tb3/socket_transport.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

namespace oracle {
namespace {

int set_nonblock(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return -1;
  }
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int set_nodelay(int fd) {
  int one = 1;
  return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

Status io_err(const char* what) {
  return Status::fail(Errc::io, std::string(what) + ": " + std::strerror(errno));
}

bool wait_fd(int fd, short events, int timeout_ms) {
  pollfd pfd{};
  pfd.fd = fd;
  pfd.events = events;
  int r = poll(&pfd, 1, timeout_ms);
  return r > 0 && (pfd.revents & events);
}

ssize_t write_full(int fd, const void* p, size_t n) {
  auto* b = static_cast<const uint8_t*>(p);
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::write(fd, b + off, n - off);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (!wait_fd(fd, POLLOUT, 5000)) {
          return -1;
        }
        continue;
      }
      return -1;
    }
    off += static_cast<size_t>(w);
  }
  return static_cast<ssize_t>(n);
}

ssize_t read_full(int fd, void* p, size_t n, int timeout_ms) {
  auto* b = static_cast<uint8_t*>(p);
  size_t off = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (off < n) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      errno = ETIMEDOUT;
      return -1;
    }
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    if (!wait_fd(fd, POLLIN, static_cast<int>(left))) {
      errno = ETIMEDOUT;
      return -1;
    }
    ssize_t r = ::read(fd, b + off, n - off);
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return -1;
    }
    if (r == 0) {
      errno = ECONNRESET;
      return -1;
    }
    off += static_cast<size_t>(r);
  }
  return static_cast<ssize_t>(n);
}

}  // namespace

TB3SocketTransport::~TB3SocketTransport() { stop(); }

void TB3SocketTransport::stop() {
  running_.store(false);
  if (listen_fd_ >= 0) {
    shutdown(listen_fd_, SHUT_RDWR);
  }
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& c : conns_) {
      if (c.fd >= 0) {
        close(c.fd);
        c.fd = -1;
      }
    }
    conns_.clear();
  }
  if (io_thread_.joinable()) {
    io_thread_.join();
  }
  if (hb_thread_.joinable()) {
    hb_thread_.join();
  }
  if (kq_ >= 0) {
    close(kq_);
    kq_ = -1;
  }
  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }
  if (hb_fd_ >= 0) {
    close(hb_fd_);
    hb_fd_ = -1;
  }
}

int TB3SocketTransport::find_fd(NodeId peer) const {
  for (const auto& c : conns_) {
    if (c.peer == peer) {
      return c.fd;
    }
  }
  return -1;
}

Status TB3SocketTransport::listen(const TransportOptions& opt) {
  opt_ = opt;
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return io_err("socket");
  }
  if (opt.reuseaddr) {
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(opt.port);
  if (inet_pton(AF_INET, opt.bind_host.c_str(), &addr.sin_addr) != 1) {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  }
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return io_err("bind");
  }
  if (::listen(fd, opt.backlog) != 0) {
    close(fd);
    return io_err("listen");
  }
  set_nonblock(fd);
  int kq = kqueue();
  if (kq < 0) {
    close(fd);
    return io_err("kqueue");
  }
  struct kevent ev {};
  EV_SET(&ev, fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
  if (kevent(kq, &ev, 1, nullptr, 0, nullptr) < 0) {
    close(kq);
    close(fd);
    return io_err("kevent add listen");
  }
  listen_fd_ = fd;
  kq_ = kq;
  auto st = LockFreeRingBuffer::create_anonymous(opt.local_ring_slots, opt.local_payload_bytes, &local_ring_);
  if (!st) {
    return st;
  }
  running_.store(true);
  io_thread_ = std::thread([this] { io_loop(); });
  return Status::OK();
}

void TB3SocketTransport::io_loop() {
  struct kevent events[16];
  while (running_.load()) {
    timespec ts{};
    ts.tv_nsec = 50 * 1000 * 1000;
    int n = kevent(kq_, nullptr, 0, events, 16, &ts);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    for (int i = 0; i < n; ++i) {
      const int fd = static_cast<int>(events[i].ident);
      // Accepts are explicit via accept_one() so tests and workers do not race the kqueue loop.
      (void)fd;
    }
  }
}

Status TB3SocketTransport::connect(NodeId peer, const std::string& host, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return io_err("socket");
  }
  if (opt_.tcp_nodelay) {
    set_nodelay(fd);
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    close(fd);
    return Status::fail(Errc::invalid_argument, "bad host");
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return io_err("connect");
  }
  set_nonblock(fd);
  std::lock_guard<std::mutex> g(mu_);
  Conn c;
  c.peer = peer;
  c.fd = fd;
  conns_.push_back(std::move(c));
  return Status::OK();
}

Status TB3SocketTransport::accept_one(NodeId expected_peer, int timeout_ms) {
  if (listen_fd_ < 0) {
    return Status::fail(Errc::invalid_argument, "not listening");
  }
  if (!wait_fd(listen_fd_, POLLIN, timeout_ms)) {
    return Status::fail(Errc::timeout, "accept timeout");
  }
  sockaddr_in peer{};
  socklen_t sl = sizeof(peer);
  int cfd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &sl);
  if (cfd < 0) {
    return io_err("accept");
  }
  if (opt_.tcp_nodelay) {
    set_nodelay(cfd);
  }
  set_nonblock(cfd);
  std::lock_guard<std::mutex> g(mu_);
  Conn c;
  c.peer = expected_peer;
  c.fd = cfd;
  conns_.push_back(std::move(c));
  return Status::OK();
}

Status TB3SocketTransport::send_fd(int fd, const TensorHeader& hdr, std::span<const std::byte> payload) {
  if (fd < 0) {
    return Status::fail(Errc::disconnected, "no connection");
  }
  if (payload.size() != hdr.nbytes) {
    return Status::fail(Errc::invalid_argument, "nbytes != payload size");
  }
  std::byte hbuf[sizeof(TensorHeader)];
  auto st = encode_header(hdr, std::span<std::byte, sizeof(TensorHeader)>(hbuf, sizeof(hbuf)));
  if (!st) {
    return st;
  }
  iovec iov[2];
  iov[0].iov_base = hbuf;
  iov[0].iov_len = sizeof(TensorHeader);
  iov[1].iov_base = const_cast<std::byte*>(payload.data());
  iov[1].iov_len = payload.size();
  int nvec = payload.empty() ? 1 : 2;
  size_t total = sizeof(TensorHeader) + payload.size();
  size_t sent = 0;
  while (sent < total) {
    if (!wait_fd(fd, POLLOUT, opt_.send_timeout_ms)) {
      return Status::fail(Errc::timeout, "send timeout");
    }
    size_t skip = sent;
    iovec cur[2];
    int nv = 0;
    size_t acc = 0;
    for (int i = 0; i < nvec; ++i) {
      if (skip >= iov[i].iov_len) {
        skip -= iov[i].iov_len;
        continue;
      }
      cur[nv].iov_base = static_cast<uint8_t*>(iov[i].iov_base) + skip;
      cur[nv].iov_len = iov[i].iov_len - skip;
      skip = 0;
      ++nv;
    }
    ssize_t w = writev(fd, cur, nv);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return io_err("writev");
    }
    sent += static_cast<size_t>(w);
  }
  (void)write_full;
  return Status::OK();
}

Status TB3SocketTransport::recv_fd(int fd, Tensor* out, int timeout_ms) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null out");
  }
  std::byte hbuf[sizeof(TensorHeader)];
  if (read_full(fd, hbuf, sizeof(hbuf), timeout_ms) < 0) {
    return Status::fail(errno == ETIMEDOUT ? Errc::timeout : Errc::disconnected, "header read");
  }
  TensorHeader hdr{};
  auto st = decode_header(std::span<const std::byte>(hbuf, sizeof(hbuf)), &hdr);
  if (!st) {
    return st;
  }
  out->header = hdr;
  out->payload.resize(static_cast<size_t>(hdr.nbytes));
  if (hdr.nbytes > 0) {
    if (read_full(fd, out->payload.data(), static_cast<size_t>(hdr.nbytes), timeout_ms) < 0) {
      return Status::fail(errno == ETIMEDOUT ? Errc::timeout : Errc::disconnected, "payload read");
    }
  }
  if (hdr.checksum != 0) {
    const uint32_t got = crc32(out->payload);
    if (got != hdr.checksum) {
      return Status::fail(Errc::protocol, "checksum mismatch");
    }
  }
  return Status::OK();
}

Status TB3SocketTransport::send_tensor(NodeId peer, const TensorHeader& hdr, std::span<const std::byte> payload) {
  std::lock_guard<std::mutex> g(mu_);
  return send_fd(find_fd(peer), hdr, payload);
}

Status TB3SocketTransport::send_tensor(NodeId peer, const Tensor& t) {
  return send_tensor(peer, t.header, t.payload);
}

Status TB3SocketTransport::recv_tensor(NodeId peer, Tensor* out, int timeout_ms) {
  std::lock_guard<std::mutex> g(mu_);
  return recv_fd(find_fd(peer), out, timeout_ms);
}

Status TB3SocketTransport::send_local(const TensorHeader& hdr, std::span<const std::byte> payload) {
  if (!local_ring_.try_push(hdr, payload)) {
    return Status::fail(Errc::busy, "local ring full");
  }
  return Status::OK();
}

Status TB3SocketTransport::recv_local(Tensor* out) {
  if (!local_ring_.try_pop_into(out)) {
    return Status::fail(Errc::not_found, "local ring empty");
  }
  return Status::OK();
}

Status TB3SocketTransport::start_heartbeat(NodeId self, const std::vector<std::pair<std::string, uint16_t>>& peers) {
  self_id_ = self;
  hb_peers_ = peers;
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    return io_err("udp socket");
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(opt_.heartbeat_port);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return io_err("udp bind");
  }
  hb_fd_ = fd;
  running_.store(true);
  hb_thread_ = std::thread([this] { hb_loop(); });
  return Status::OK();
}

void TB3SocketTransport::hb_loop() {
  uint8_t buf[16];
  while (running_.load()) {
    std::memcpy(buf, &kTensorMagic, 4);
    std::memcpy(buf + 4, &self_id_, 4);
    for (const auto& [host, port] : hb_peers_) {
      sockaddr_in a{};
      a.sin_family = AF_INET;
      a.sin_port = htons(port);
      inet_pton(AF_INET, host.c_str(), &a.sin_addr);
      sendto(hb_fd_, buf, 8, 0, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    }
    pollfd pfd{hb_fd_, POLLIN, 0};
    if (poll(&pfd, 1, 200) > 0) {
      sockaddr_in from{};
      socklen_t sl = sizeof(from);
      uint8_t rbuf[16];
      ssize_t n = recvfrom(hb_fd_, rbuf, sizeof(rbuf), 0, reinterpret_cast<sockaddr*>(&from), &sl);
      if (n >= 8 && handler_) {
        uint32_t magic = 0;
        NodeId id = 0;
        std::memcpy(&magic, rbuf, 4);
        std::memcpy(&id, rbuf + 4, 4);
        if (magic == kTensorMagic) {
          Tensor t;
          t.header.flags = kFlagHeartbeat;
          t.header.seq_id = id;
          handler_(id, std::move(t));
        }
      }
    }
  }
}

}  // namespace oracle

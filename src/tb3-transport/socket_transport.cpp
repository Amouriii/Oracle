#include "oracle/tb3/socket_transport.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

namespace oracle {
namespace {

constexpr const char* kHelloMagic = "ORACLE-HELLO";
constexpr const char* kAckOk = "OK";

int set_nonblock(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return -1;
  }
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

Status io_err(const char* what) {
  return Status::fail(Errc::io, std::string(what) + ": " + std::strerror(errno));
}

bool wait_fd(int fd, short events, int timeout_ms) {
  pollfd pfd{};
  pfd.fd = fd;
  pfd.events = events;
  const int r = poll(&pfd, 1, timeout_ms);
  return r > 0 && (pfd.revents & events) != 0;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string next_line(std::string_view& s) {
  const auto nl = s.find('\n');
  if (nl == std::string_view::npos) {
    const std::string out(s);
    s = {};
    return out;
  }
  const std::string out(s.substr(0, nl));
  s.remove_prefix(nl + 1);
  return out;
}

}  // namespace

std::string Handshake::encode() const {
  std::ostringstream os;
  os << kHelloMagic << "\n"
     << node_id << "\n"
     << role << "\n"
     << protocol_version << "\n"
     << nonce << "\n"
     << signature << "\n"
     << runner << "\n"
     << layer_start << "\n"
     << layer_end << "\n";
  return os.str();
}

Status Handshake::decode(std::string_view blob, Handshake* out) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "handshake: null out");
  }
  std::string_view s = blob;
  if (next_line(s) != kHelloMagic) {
    return Status::fail(Errc::protocol, "handshake: bad magic");
  }
  Handshake h;
  try {
    h.node_id = static_cast<NodeId>(std::stoul(next_line(s)));
    h.role = next_line(s);
    h.protocol_version = static_cast<uint32_t>(std::stoul(next_line(s)));
    h.nonce = next_line(s);
    h.signature = next_line(s);
    h.runner = next_line(s);
    h.layer_start = static_cast<uint32_t>(std::stoul(next_line(s)));
    h.layer_end = static_cast<uint32_t>(std::stoul(next_line(s)));
  } catch (const std::exception&) {
    return Status::fail(Errc::protocol, "handshake: malformed field");
  }
  if (h.protocol_version != kProtocolVersion) {
    return Status::fail(Errc::protocol, "handshake: protocol version " +
                                            std::to_string(h.protocol_version) + " != " +
                                            std::to_string(kProtocolVersion));
  }
  *out = std::move(h);
  return Status::OK();
}

TB3SocketTransport::~TB3SocketTransport() { stop(); }

void TB3SocketTransport::set_recv_handler(RecvHandler h) {
  std::lock_guard<std::mutex> g(mu_);
  handler_ = std::move(h);
}

void TB3SocketTransport::set_auth_handler(AuthHandler h) {
  std::lock_guard<std::mutex> g(mu_);
  auth_ = std::move(h);
}

void TB3SocketTransport::set_handshake(Handshake h) {
  std::lock_guard<std::mutex> g(mu_);
  hello_ = std::move(h);
}

void TB3SocketTransport::set_heartbeat_payload(std::string payload) {
  std::lock_guard<std::mutex> g(hb_mu_);
  hb_payload_ = std::move(payload);
}

void TB3SocketTransport::stop() {
  running_.store(false);
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& [id, c] : conns_) {
      if (c && c->fd >= 0) {
        ::shutdown(c->fd, SHUT_RDWR);
        ::close(c->fd);
        c->fd = -1;
      }
    }
    conns_.clear();
  }
  if (hb_thread_.joinable()) {
    hb_thread_.join();
  }
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (hb_fd_ >= 0) {
    ::close(hb_fd_);
    hb_fd_ = -1;
  }
}

void TB3SocketTransport::disconnect(NodeId peer) {
  std::shared_ptr<Conn> c;
  {
    std::lock_guard<std::mutex> g(mu_);
    const auto it = conns_.find(peer);
    if (it == conns_.end()) {
      return;
    }
    c = it->second;
    conns_.erase(it);
  }
  if (c && c->fd >= 0) {
    ::shutdown(c->fd, SHUT_RDWR);
    ::close(c->fd);
    c->fd = -1;
  }
}

std::shared_ptr<TB3SocketTransport::Conn> TB3SocketTransport::find(NodeId peer) const {
  std::lock_guard<std::mutex> g(mu_);
  const auto it = conns_.find(peer);
  return it == conns_.end() ? nullptr : it->second;
}

std::shared_ptr<TB3SocketTransport::Conn> TB3SocketTransport::adopt(NodeId peer, int fd,
                                                                    std::string host, uint16_t port) {
  auto c = std::make_shared<Conn>();
  c->peer = peer;
  c->fd = fd;
  c->host = std::move(host);
  c->port = port;

  std::shared_ptr<Conn> old;
  {
    std::lock_guard<std::mutex> g(mu_);
    const auto it = conns_.find(peer);
    if (it != conns_.end()) {
      old = it->second;
      c->reconnects.store(old->reconnects.load() + 1);
      c->bytes_sent.store(old->bytes_sent.load());
      c->bytes_recv.store(old->bytes_recv.load());
      c->frames_sent.store(old->frames_sent.load());
      c->frames_recv.store(old->frames_recv.load());
      c->errors.store(old->errors.load());
    }
    conns_[peer] = c;
  }
  if (old && old->fd >= 0) {
    ::shutdown(old->fd, SHUT_RDWR);
    ::close(old->fd);
    old->fd = -1;
    reconnects_.fetch_add(1, std::memory_order_relaxed);
  }
  return c;
}

void TB3SocketTransport::tune_socket(int fd) const {
  if (opt_.tcp_nodelay) {
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }
  if (opt_.send_buffer_bytes > 0) {
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &opt_.send_buffer_bytes, sizeof(opt_.send_buffer_bytes));
  }
  if (opt_.recv_buffer_bytes > 0) {
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &opt_.recv_buffer_bytes, sizeof(opt_.recv_buffer_bytes));
  }
#ifdef SO_KEEPALIVE
  const int keep = 1;
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keep, sizeof(keep));
#endif
  set_nonblock(fd);
}

Status TB3SocketTransport::listen(const TransportOptions& opt) {
  opt_ = opt;
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return io_err("socket");
  }
  if (opt.reuseaddr) {
    const int one = 1;
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
    const auto st = io_err("bind");
    ::close(fd);
    return st;
  }
  if (::listen(fd, opt.backlog) != 0) {
    const auto st = io_err("listen");
    ::close(fd);
    return st;
  }
  set_nonblock(fd);
  listen_fd_ = fd;

  auto st = LockFreeRingBuffer::create_anonymous(opt.local_ring_slots, opt.local_payload_bytes,
                                                &local_ring_);
  if (!st) {
    ::close(fd);
    listen_fd_ = -1;
    return st;
  }
  running_.store(true);
  return Status::OK();
}

Status TB3SocketTransport::connect(NodeId peer, const std::string& host, uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return io_err("socket");
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    ::close(fd);
    return Status::fail(Errc::invalid_argument, "not a dotted-quad address: " + host);
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    const auto st = io_err("connect");
    ::close(fd);
    return st;
  }
  tune_socket(fd);
  adopt(peer, fd, host, port);
  return Status::OK();
}

Status TB3SocketTransport::ensure_connected(NodeId peer, const std::string& host, uint16_t port,
                                            int timeout_ms) {
  if (connected(peer)) {
    return Status::OK();
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  int backoff_ms = 50;
  Status last = Status::fail(Errc::disconnected, "not attempted");
  for (;;) {
    last = connect(peer, host, port);
    if (last) {
      return last;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return Status::fail(Errc::timeout, "could not reach node " + std::to_string(peer) + " at " + host +
                                             ":" + std::to_string(port) + " (" + last.message + ")");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
    backoff_ms = std::min(backoff_ms * 2, 1000);
  }
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
  const int cfd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &sl);
  if (cfd < 0) {
    return io_err("accept");
  }
  tune_socket(cfd);
  char ip[INET_ADDRSTRLEN]{};
  inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));
  adopt(expected_peer, cfd, ip, ntohs(peer.sin_port));
  return Status::OK();
}

Status TB3SocketTransport::accept_registration(int timeout_ms, Handshake* out) {
  if (listen_fd_ < 0) {
    return Status::fail(Errc::invalid_argument, "not listening");
  }
  if (!wait_fd(listen_fd_, POLLIN, timeout_ms)) {
    return Status::fail(Errc::timeout, "no node tried to register");
  }
  sockaddr_in peer{};
  socklen_t sl = sizeof(peer);
  const int cfd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &sl);
  if (cfd < 0) {
    return io_err("accept");
  }
  tune_socket(cfd);
  char ip[INET_ADDRSTRLEN]{};
  inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));

  // Read the hello on a temporary connection so an unauthenticated peer is
  // never registered under a node id it has not proved it owns.
  Conn tmp;
  tmp.fd = cfd;
  tmp.host = ip;
  tmp.port = ntohs(peer.sin_port);
  Tensor frame;
  auto st = recv_fd(tmp, &frame, timeout_ms > 0 ? timeout_ms : 5000);
  if (!st) {
    ::close(cfd);
    return st;
  }
  if ((frame.header.flags & kFlagControl) == 0) {
    ::close(cfd);
    return Status::fail(Errc::protocol, "first frame from a joining node was not a handshake");
  }
  Handshake h;
  st = Handshake::decode(
      std::string_view(reinterpret_cast<const char*>(frame.payload.data()), frame.payload.size()), &h);
  if (!st) {
    ::close(cfd);
    return st;
  }

  AuthHandler auth;
  {
    std::lock_guard<std::mutex> g(mu_);
    auth = auth_;
  }
  std::string reason;
  const bool ok = auth ? auth(h, &reason) : true;

  TensorHeader ack{};
  ack.flags = kFlagControl | kFlagAck;
  const std::string body = ok ? std::string(kAckOk) : ("DENIED: " + (reason.empty() ? "rejected" : reason));
  ack.dtype = static_cast<uint16_t>(DType::I8);
  ack.rank = 1;
  ack.shape[0] = static_cast<uint32_t>(body.size());
  ack.nbytes = body.size();
  auto payload = std::span<const std::byte>(reinterpret_cast<const std::byte*>(body.data()), body.size());
  ack.checksum = crc32(payload);
  (void)send_fd(tmp, ack, payload);

  if (!ok) {
    ::close(cfd);
    return Status::fail(Errc::invalid_argument, "registration refused: " + reason);
  }
  tmp.fd = -1;  // ownership moves to the registry below
  adopt(h.node_id, cfd, ip, ntohs(peer.sin_port));
  if (out) {
    *out = h;
  }
  return Status::OK();
}

Status TB3SocketTransport::register_with(NodeId peer, const Handshake& hello, int timeout_ms) {
  auto c = find(peer);
  if (!c) {
    return Status::fail(Errc::disconnected, "no connection to node " + std::to_string(peer));
  }
  const std::string blob = hello.encode();
  TensorHeader hdr{};
  hdr.flags = kFlagControl;
  hdr.dtype = static_cast<uint16_t>(DType::I8);
  hdr.rank = 1;
  hdr.shape[0] = static_cast<uint32_t>(blob.size());
  hdr.nbytes = blob.size();
  auto payload = std::span<const std::byte>(reinterpret_cast<const std::byte*>(blob.data()), blob.size());
  hdr.checksum = crc32(payload);
  auto st = send_tensor(peer, hdr, payload);
  if (!st) {
    return st;
  }
  Tensor ack;
  st = recv_tensor(peer, &ack, timeout_ms);
  if (!st) {
    return st;
  }
  const std::string body(reinterpret_cast<const char*>(ack.payload.data()), ack.payload.size());
  if ((ack.header.flags & kFlagAck) == 0 || body != kAckOk) {
    return Status::fail(Errc::invalid_argument,
                        body.empty() ? "registration refused by node " + std::to_string(peer) : body);
  }
  return Status::OK();
}

Status TB3SocketTransport::send_fd(Conn& c, const TensorHeader& hdr, std::span<const std::byte> payload) {
  if (c.fd < 0) {
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
  const auto t0 = std::chrono::steady_clock::now();
  iovec iov[2];
  iov[0].iov_base = hbuf;
  iov[0].iov_len = sizeof(TensorHeader);
  iov[1].iov_base = const_cast<std::byte*>(payload.data());
  iov[1].iov_len = payload.size();
  const int nvec = payload.empty() ? 1 : 2;
  const size_t total = sizeof(TensorHeader) + payload.size();
  size_t sent = 0;
  while (sent < total) {
    // Rebuild the iovec from the current offset: writev is free to make partial
    // progress, and header + body must go out as one contiguous byte stream.
    iovec cur[2];
    int nv = 0;
    size_t skip = sent;
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
    const ssize_t w = writev(c.fd, cur, nv);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (!wait_fd(c.fd, POLLOUT, opt_.send_timeout_ms)) {
          c.errors.fetch_add(1, std::memory_order_relaxed);
          return Status::fail(Errc::timeout, "send timeout");
        }
        continue;
      }
      c.errors.fetch_add(1, std::memory_order_relaxed);
      return io_err("writev");
    }
    sent += static_cast<size_t>(w);
  }
  c.bytes_sent.fetch_add(total, std::memory_order_relaxed);
  c.frames_sent.fetch_add(1, std::memory_order_relaxed);
  c.last_send_ms.store(ms_since(t0), std::memory_order_relaxed);
  return Status::OK();
}

Status TB3SocketTransport::recv_fd(Conn& c, Tensor* out, int timeout_ms) {
  if (!out) {
    return Status::fail(Errc::invalid_argument, "null out");
  }
  if (c.fd < 0) {
    return Status::fail(Errc::disconnected, "no connection");
  }
  const auto t0 = std::chrono::steady_clock::now();
  const auto deadline = t0 + std::chrono::milliseconds(timeout_ms);

  auto read_exact = [&](void* dst, size_t n) -> Status {
    auto* b = static_cast<uint8_t*>(dst);
    size_t off = 0;
    while (off < n) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return Status::fail(Errc::timeout, "read timeout");
      }
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
      if (!wait_fd(c.fd, POLLIN, static_cast<int>(left))) {
        return Status::fail(Errc::timeout, "read timeout");
      }
      const ssize_t r = ::read(c.fd, b + off, n - off);
      if (r < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
          continue;
        }
        return io_err("read");
      }
      if (r == 0) {
        return Status::fail(Errc::disconnected, "peer closed the connection");
      }
      off += static_cast<size_t>(r);
    }
    return Status::OK();
  };

  std::byte hbuf[sizeof(TensorHeader)];
  auto st = read_exact(hbuf, sizeof(hbuf));
  if (!st) {
    if (st.code != Errc::timeout) {
      c.errors.fetch_add(1, std::memory_order_relaxed);
    }
    return st;
  }
  TensorHeader hdr{};
  st = decode_header(std::span<const std::byte>(hbuf, sizeof(hbuf)), &hdr);
  if (!st) {
    c.errors.fetch_add(1, std::memory_order_relaxed);
    return st;
  }
  if (opt_.max_payload_bytes && hdr.nbytes > opt_.max_payload_bytes) {
    // Refuse before allocating: a bogus length header is otherwise a one-packet
    // memory exhaustion attack.
    c.errors.fetch_add(1, std::memory_order_relaxed);
    return Status::fail(Errc::protocol, "frame declares " + std::to_string(hdr.nbytes) +
                                            " payload bytes, over the " +
                                            std::to_string(opt_.max_payload_bytes) + " byte limit");
  }
  out->header = hdr;
  try {
    out->payload.resize(static_cast<size_t>(hdr.nbytes));
  } catch (const std::bad_alloc&) {
    return Status::fail(Errc::oom, "cannot allocate " + std::to_string(hdr.nbytes) + " payload bytes");
  }
  if (hdr.nbytes > 0) {
    st = read_exact(out->payload.data(), static_cast<size_t>(hdr.nbytes));
    if (!st) {
      if (st.code != Errc::timeout) {
        c.errors.fetch_add(1, std::memory_order_relaxed);
      }
      return st;
    }
  }
  if (hdr.checksum != 0 && crc32(out->payload) != hdr.checksum) {
    c.errors.fetch_add(1, std::memory_order_relaxed);
    return Status::fail(Errc::protocol, "payload checksum mismatch");
  }
  c.bytes_recv.fetch_add(sizeof(TensorHeader) + out->payload.size(), std::memory_order_relaxed);
  c.frames_recv.fetch_add(1, std::memory_order_relaxed);
  c.last_recv_ms.store(ms_since(t0), std::memory_order_relaxed);
  return Status::OK();
}

Status TB3SocketTransport::send_tensor(NodeId peer, const TensorHeader& hdr,
                                       std::span<const std::byte> payload) {
  auto c = find(peer);
  if (!c) {
    return Status::fail(Errc::disconnected, "no connection to node " + std::to_string(peer));
  }
  std::lock_guard<std::mutex> g(c->send_mu);
  return send_fd(*c, hdr, payload);
}

Status TB3SocketTransport::send_tensor(NodeId peer, const Tensor& t) {
  return send_tensor(peer, t.header, t.payload);
}

Status TB3SocketTransport::recv_tensor(NodeId peer, Tensor* out, int timeout_ms) {
  auto c = find(peer);
  if (!c) {
    return Status::fail(Errc::disconnected, "no connection to node " + std::to_string(peer));
  }
  std::lock_guard<std::mutex> g(c->recv_mu);
  return recv_fd(*c, out, timeout_ms);
}

Status TB3SocketTransport::recv_any(NodeId* from, Tensor* out, int timeout_ms) {
  std::vector<std::shared_ptr<Conn>> live;
  {
    std::lock_guard<std::mutex> g(mu_);
    live.reserve(conns_.size());
    for (auto& [id, c] : conns_) {
      if (c && c->fd >= 0) {
        live.push_back(c);
      }
    }
  }
  if (live.empty()) {
    return Status::fail(Errc::disconnected, "no connected peers");
  }
  std::vector<pollfd> pfds;
  pfds.reserve(live.size());
  for (const auto& c : live) {
    pfds.push_back(pollfd{c->fd, POLLIN, 0});
  }
  const int r = poll(pfds.data(), pfds.size(), timeout_ms);
  if (r < 0) {
    return io_err("poll");
  }
  if (r == 0) {
    return Status::fail(Errc::timeout, "no frame arrived");
  }
  for (size_t i = 0; i < pfds.size(); ++i) {
    if ((pfds[i].revents & POLLIN) == 0) {
      continue;
    }
    if (from) {
      *from = live[i]->peer;
    }
    std::lock_guard<std::mutex> g(live[i]->recv_mu);
    return recv_fd(*live[i], out, timeout_ms);
  }
  return Status::fail(Errc::timeout, "no readable peer");
}

Status TB3SocketTransport::send_local(const TensorHeader& hdr, std::span<const std::byte> payload) {
  if (!local_ring_.try_push(hdr, payload)) {
    return Status::fail(Errc::busy, "local ring full");
  }
  local_frames_.fetch_add(1, std::memory_order_relaxed);
  return Status::OK();
}

Status TB3SocketTransport::recv_local(Tensor* out) {
  if (!local_ring_.try_pop_into(out)) {
    return Status::fail(Errc::not_found, "local ring empty");
  }
  return Status::OK();
}

bool TB3SocketTransport::connected(NodeId peer) const {
  const auto c = find(peer);
  return c && c->fd >= 0;
}

std::vector<PeerStats> TB3SocketTransport::peers() const {
  std::vector<std::shared_ptr<Conn>> live;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (const auto& [id, c] : conns_) {
      live.push_back(c);
    }
  }
  std::vector<PeerStats> out;
  out.reserve(live.size());
  for (const auto& c : live) {
    PeerStats p;
    p.id = c->peer;
    p.host = c->host;
    p.port = c->port;
    p.connected = c->fd >= 0;
    p.bytes_sent = c->bytes_sent.load();
    p.bytes_recv = c->bytes_recv.load();
    p.frames_sent = c->frames_sent.load();
    p.frames_recv = c->frames_recv.load();
    p.reconnects = c->reconnects.load();
    p.errors = c->errors.load();
    p.last_send_ms = c->last_send_ms.load();
    p.last_recv_ms = c->last_recv_ms.load();
    out.push_back(std::move(p));
  }
  std::sort(out.begin(), out.end(), [](const PeerStats& a, const PeerStats& b) { return a.id < b.id; });
  return out;
}

TransportTotals TB3SocketTransport::totals() const {
  TransportTotals t;
  for (const auto& p : peers()) {
    t.bytes_sent += p.bytes_sent;
    t.bytes_recv += p.bytes_recv;
    t.frames_sent += p.frames_sent;
    t.frames_recv += p.frames_recv;
    t.reconnects += p.reconnects;
    t.errors += p.errors;
    ++t.peers;
    if (p.connected) {
      ++t.connected_peers;
    }
  }
  t.local_frames = local_frames_.load();
  return t;
}

Status TB3SocketTransport::start_heartbeat(NodeId self,
                                           const std::vector<std::pair<std::string, uint16_t>>& peers) {
  self_id_ = self;
  hb_peers_ = peers;
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    return io_err("udp socket");
  }
  const int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(opt_.heartbeat_port);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    const auto st = io_err("udp bind");
    ::close(fd);
    return st;
  }
  hb_fd_ = fd;
  running_.store(true);
  hb_thread_ = std::thread([this] { hb_loop(); });
  return Status::OK();
}

void TB3SocketTransport::hb_loop() {
  // Datagram layout: magic | node id | payload (a short status blob the master
  // folds into worker load tracking).
  while (running_.load()) {
    std::string payload;
    {
      std::lock_guard<std::mutex> g(hb_mu_);
      payload = hb_payload_;
    }
    std::vector<uint8_t> buf(8 + payload.size());
    std::memcpy(buf.data(), &kTensorMagic, 4);
    std::memcpy(buf.data() + 4, &self_id_, 4);
    if (!payload.empty()) {
      std::memcpy(buf.data() + 8, payload.data(), payload.size());
    }
    for (const auto& [host, port] : hb_peers_) {
      sockaddr_in a{};
      a.sin_family = AF_INET;
      a.sin_port = htons(port);
      if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) {
        continue;
      }
      sendto(hb_fd_, buf.data(), buf.size(), 0, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    }

    pollfd pfd{hb_fd_, POLLIN, 0};
    if (poll(&pfd, 1, 200) > 0) {
      sockaddr_in from{};
      socklen_t sl = sizeof(from);
      uint8_t rbuf[1500];
      const ssize_t n = recvfrom(hb_fd_, rbuf, sizeof(rbuf), 0, reinterpret_cast<sockaddr*>(&from), &sl);
      if (n >= 8) {
        uint32_t magic = 0;
        NodeId id = 0;
        std::memcpy(&magic, rbuf, 4);
        std::memcpy(&id, rbuf + 4, 4);
        RecvHandler handler;
        {
          std::lock_guard<std::mutex> g(mu_);
          handler = handler_;
        }
        if (magic == kTensorMagic && handler) {
          Tensor t;
          t.header.flags = kFlagHeartbeat;
          t.header.seq_id = id;
          t.payload.assign(reinterpret_cast<const std::byte*>(rbuf + 8),
                           reinterpret_cast<const std::byte*>(rbuf + n));
          t.header.nbytes = t.payload.size();
          handler(id, std::move(t));
        }
      }
    }
  }
}

}  // namespace oracle

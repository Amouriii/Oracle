#include "oracle/tb3/socket_transport.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  const uint16_t port = 19200;
  std::atomic<bool> ok{false};
  std::string err;

  std::thread server([&] {
    oracle::TB3SocketTransport tx;
    oracle::TransportOptions opt;
    opt.port = port;
    opt.bind_host = "127.0.0.1";
    auto st = tx.listen(opt);
    if (!st) {
      err = st.message;
      return;
    }
    st = tx.accept_one(1, 5000);
    if (!st) {
      err = st.message;
      return;
    }
    oracle::Tensor t;
    st = tx.recv_tensor(1, &t, 5000);
    if (!st) {
      err = st.message;
      return;
    }
    st = tx.send_tensor(1, t);
    if (!st) {
      err = st.message;
      return;
    }
    ok = true;
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  oracle::TB3SocketTransport client;
  auto st = client.connect(0, "127.0.0.1", port);
  if (!st) {
    server.join();
    std::cerr << "connect " << st.message << "\n";
    return 1;
  }
  std::vector<std::byte> payload(64);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::byte>(i);
  }
  oracle::TensorHeader hdr;
  hdr.dtype = static_cast<uint16_t>(oracle::DType::F16);
  hdr.rank = 1;
  hdr.shape[0] = 32;
  hdr.nbytes = payload.size();
  hdr.seq_id = 7;
  hdr.flags = oracle::kFlagDecode;
  hdr.checksum = oracle::crc32(payload);
  st = client.send_tensor(0, hdr, payload);
  if (!st) {
    server.join();
    std::cerr << "send " << st.message << "\n";
    return 1;
  }
  oracle::Tensor echo;
  st = client.recv_tensor(0, &echo, 5000);
  server.join();
  if (!st) {
    std::cerr << "recv " << st.message << " err=" << err << "\n";
    return 1;
  }
  assert(echo.header.seq_id == 7);
  assert(echo.payload == payload);
  if (!ok && err.size()) {
    std::cerr << err << "\n";
    return 1;
  }
  std::cout << "test_transport_tcp ok\n";
  return 0;
}

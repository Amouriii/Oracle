#include "oracle/tb3/socket_transport.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  uint16_t port = 9200;
  bool server = false;
  int iters = 200;
  uint32_t bytes = 8192 * 2;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--listen") == 0) {
      server = true;
    } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
      host = argv[++i];
    } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = static_cast<uint16_t>(std::stoul(argv[++i]));
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      bytes = static_cast<uint32_t>(std::stoul(argv[++i]));
    }
  }

  oracle::TB3SocketTransport tx;
  oracle::TransportOptions opt;
  opt.port = port;
  opt.bind_host = "0.0.0.0";
  opt.local_payload_bytes = bytes;
  if (server) {
    auto st = tx.listen(opt);
    if (!st) {
      std::cerr << st.message << "\n";
      return 1;
    }
    st = tx.accept_one(1, 60000);
    if (!st) {
      std::cerr << st.message << "\n";
      return 1;
    }
    for (int i = 0; i < iters; ++i) {
      oracle::Tensor t;
      st = tx.recv_tensor(1, &t, 10000);
      if (!st) {
        std::cerr << "recv " << st.message << "\n";
        return 1;
      }
      st = tx.send_tensor(1, t);
      if (!st) {
        return 1;
      }
    }
    return 0;
  }

  auto st = tx.connect(0, host, port);
  if (!st) {
    std::cerr << st.message << "\n";
    return 1;
  }
  std::vector<std::byte> payload(bytes);
  for (uint32_t i = 0; i < bytes; ++i) {
    payload[i] = static_cast<std::byte>(i & 0xFF);
  }
  oracle::TensorHeader hdr;
  hdr.dtype = static_cast<uint16_t>(oracle::DType::F16);
  hdr.rank = 1;
  hdr.shape[0] = bytes / 2;
  hdr.nbytes = bytes;
  hdr.checksum = oracle::crc32(payload);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    hdr.seq_id = static_cast<uint64_t>(i);
    st = tx.send_tensor(0, hdr, payload);
    if (!st) {
      std::cerr << st.message << "\n";
      return 1;
    }
    oracle::Tensor echo;
    st = tx.recv_tensor(0, &echo, 10000);
    if (!st) {
      std::cerr << st.message << "\n";
      return 1;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();
  const double bits = static_cast<double>(bytes) * 2.0 * iters * 8.0;  // send+echo
  const double gbps = (bits / sec) / 1e9;
  const double rtt_ms = (sec / iters) * 1000.0;
  std::cout << "iters=" << iters << " bytes=" << bytes << " gbps_bidir=" << gbps << " rtt_ms=" << rtt_ms << "\n";
  std::cout << "gate: >8 Gbps TB3 TCP (lab), decode RTT << 5 ms\n";
  return 0;
}

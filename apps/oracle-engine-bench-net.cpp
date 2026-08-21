// Measures what the activation path between two nodes can actually do:
// round-trip latency at decode-sized payloads, one-way throughput at
// prefill-sized payloads, and the transfer time for one hidden-state frame.
//
//   node A:  oracle-engine-bench-net --listen --port 9200
//   node B:  oracle-engine-bench-net --host 10.10.0.1 --port 9200
#include "oracle/tb3/socket_transport.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

double percentile(std::vector<double> v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(v.size() - 1, lo + 1);
  const double frac = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

std::string human_rate(double gbps) {
  char buf[64];
  if (gbps >= 1.0) {
    std::snprintf(buf, sizeof(buf), "%.2f Gb/s", gbps);
  } else {
    std::snprintf(buf, sizeof(buf), "%.1f Mb/s", gbps * 1000.0);
  }
  return buf;
}

void usage() {
  std::cout <<
      "oracle-engine-bench-net [--listen] [--host H] [--port P] [--iters N] [--bytes N]\n"
      "  --listen        act as the echo side of the benchmark\n"
      "  --host H        peer address (client side)\n"
      "  --port P        transport port (default 9200)\n"
      "  --iters N       round trips per size (default 200)\n"
      "  --bytes N       benchmark this size only, instead of the default sweep\n"
      "  --hidden N      hidden dimension used to label decode-sized frames\n";
}

struct SizeResult {
  uint32_t bytes{0};
  double rtt_p50{0};
  double rtt_p99{0};
  double rtt_min{0};
  double one_way_gbps{0};
  double round_trip_gbps{0};
};

}  // namespace

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  uint16_t port = 9200;
  bool server = false;
  int iters = 200;
  uint32_t only_bytes = 0;
  uint32_t hidden = 8192;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--listen") {
      server = true;
    } else if (a == "--host" && i + 1 < argc) {
      host = argv[++i];
    } else if (a == "--port" && i + 1 < argc) {
      port = static_cast<uint16_t>(std::stoul(argv[++i]));
    } else if (a == "--iters" && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else if (a == "--bytes" && i + 1 < argc) {
      only_bytes = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if (a == "--hidden" && i + 1 < argc) {
      hidden = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    }
  }

  // A decode step ships one hidden state; a prefill ships the whole prompt.
  std::vector<uint32_t> sizes;
  if (only_bytes) {
    sizes.push_back(only_bytes);
  } else {
    sizes = {hidden * 2, hidden * 2 * 8, hidden * 2 * 64, 1u << 20, 8u << 20};
  }

  oracle::TB3SocketTransport tx;
  oracle::TransportOptions opt;
  opt.port = port;
  opt.bind_host = "0.0.0.0";
  opt.local_payload_bytes = sizes.back();
  opt.max_payload_bytes = 64ull << 20;
  opt.send_timeout_ms = 30000;
  opt.recv_timeout_ms = 30000;

  if (server) {
    auto st = tx.listen(opt);
    if (!st) {
      std::cerr << st.message << "\n";
      return 1;
    }
    std::cout << "bench: listening on :" << port << ", echoing frames back\n";
    st = tx.accept_one(1, 120000);
    if (!st) {
      std::cerr << st.message << "\n";
      return 1;
    }
    for (;;) {
      oracle::Tensor t;
      st = tx.recv_tensor(1, &t, 120000);
      if (!st) {
        if (st.code == oracle::Errc::disconnected) {
          std::cout << "bench: peer finished\n";
          return 0;
        }
        std::cerr << "recv: " << st.message << "\n";
        return 1;
      }
      st = tx.send_tensor(1, t);
      if (!st) {
        std::cerr << "send: " << st.message << "\n";
        return 1;
      }
    }
  }

  auto st = tx.connect(0, host, port);
  if (!st) {
    std::cerr << "connect " << host << ":" << port << ": " << st.message << "\n";
    return 1;
  }
  std::cout << "bench: " << host << ":" << port << ", " << iters << " round trips per size\n\n";
  std::cout << std::left << std::setw(12) << "payload" << std::setw(12) << "rtt p50"
            << std::setw(12) << "rtt p99" << std::setw(12) << "rtt min" << std::setw(14) << "one-way"
            << "round-trip\n";

  std::vector<SizeResult> results;
  for (uint32_t bytes : sizes) {
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

    // A few untimed round trips first so TCP's window has grown before we
    // start recording; otherwise slow start dominates the large sizes.
    for (int i = 0; i < 5; ++i) {
      hdr.seq_id = 0;
      if (!tx.send_tensor(0, hdr, payload)) {
        break;
      }
      oracle::Tensor echo;
      (void)tx.recv_tensor(0, &echo, 30000);
    }

    std::vector<double> rtts;
    rtts.reserve(static_cast<size_t>(iters));
    const auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      hdr.seq_id = static_cast<uint64_t>(i);
      const auto t0 = std::chrono::steady_clock::now();
      st = tx.send_tensor(0, hdr, payload);
      if (!st) {
        std::cerr << "\nsend: " << st.message << "\n";
        return 1;
      }
      oracle::Tensor echo;
      st = tx.recv_tensor(0, &echo, 30000);
      if (!st) {
        std::cerr << "\nrecv: " << st.message << "\n";
        return 1;
      }
      rtts.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                         .count());
    }
    const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                                 .count();

    SizeResult r;
    r.bytes = bytes;
    r.rtt_p50 = percentile(rtts, 0.50);
    r.rtt_p99 = percentile(rtts, 0.99);
    r.rtt_min = rtts.empty() ? 0 : *std::min_element(rtts.begin(), rtts.end());
    const double bits = static_cast<double>(bytes) * 8.0 * iters;
    r.one_way_gbps = (bits / elapsed_s) / 1e9;
    r.round_trip_gbps = (2.0 * bits / elapsed_s) / 1e9;
    results.push_back(r);

    std::ostringstream label;
    label << (bytes >= (1u << 20) ? std::to_string(bytes >> 20) + " MiB"
                                  : std::to_string(bytes >> 10) + " KiB");
    std::cout << std::left << std::setw(12) << label.str() << std::fixed << std::setprecision(3)
              << std::setw(12) << r.rtt_p50 << std::setw(12) << r.rtt_p99 << std::setw(12) << r.rtt_min
              << std::setw(14) << human_rate(r.one_way_gbps) << human_rate(r.round_trip_gbps) << "\n";
  }

  const auto& decode = results.front();
  const auto& bulk = results.back();
  std::cout << "\ndecode hop  " << std::fixed << std::setprecision(3) << decode.rtt_p50
            << " ms per round trip at " << (decode.bytes >> 10) << " KiB"
            << "  (one activation of a " << hidden << "-wide model in f16)\n";
  std::cout << "bulk        " << human_rate(bulk.one_way_gbps) << " one-way at "
            << (bulk.bytes >> 20) << " MiB\n";
  std::cout << "\ngates: Thunderbolt 3 bridge should clear 8 Gb/s bulk and keep the decode hop\n"
               "       well under 5 ms; a 1 Gb ethernet link will show ~0.9 Gb/s.\n";
  const bool bulk_ok = bulk.one_way_gbps >= 8.0;
  const bool rtt_ok = decode.rtt_p50 < 5.0;
  std::cout << "result: bulk " << (bulk_ok ? "PASS" : "below gate") << ", decode rtt "
            << (rtt_ok ? "PASS" : "above gate") << "\n";
  return 0;
}

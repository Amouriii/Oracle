#!/usr/bin/env bash
set -euo pipefail
# Phase 1: Thunderbolt Bridge throughput + engine tensor ping-pong.
# Usage: scripts/phase1_net_bench.sh [peer_ip]
PEER="${1:-10.10.0.2}"
PORT="${2:-9200}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/oracle-engine-bench-net"

echo "== iperf3 TCP to ${PEER} (install iperf3 if missing) =="
if command -v iperf3 >/dev/null 2>&1; then
  iperf3 -c "${PEER}" -t 8 || true
else
  echo "iperf3 not found; skip. Gate: >8 Gbps on TB3 Bridge."
fi

echo "== oracle-engine-bench-net client -> ${PEER}:${PORT} =="
if [[ ! -x "${BIN}" ]]; then
  echo "build first: cmake -S . -B build && cmake --build build"
  echo "on peer: ${BIN} --listen --port ${PORT}"
  exit 1
fi
"${BIN}" --host "${PEER}" --port "${PORT}" --iters 400 --bytes 16384
echo "Decode RTT gate: << 5 ms (hidden_dim F16 ~16 KiB)."

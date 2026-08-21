#!/usr/bin/env bash
# Phase 1 - network: is the link between two nodes good enough to carry
# activations?  Run the peer side first:
#
#   peer:  ./build/oracle-engine-bench-net --listen --port 9200
#   here:  scripts/phase1_net_bench.sh 10.10.0.2
set -euo pipefail
PEER="${1:-10.10.0.2}"
PORT="${2:-9200}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/oracle-engine-bench-net"

if [[ ! -x "${BIN}" ]]; then
  "${ROOT}/scripts/build.sh"
fi

echo "== iperf3 baseline to ${PEER} =="
if command -v iperf3 >/dev/null 2>&1; then
  iperf3 -c "${PEER}" -t 8 || echo "(iperf3 failed; is the server running there?)"
else
  echo "iperf3 not installed; skipping the independent baseline"
fi

echo
echo "== Oracle tensor path to ${PEER}:${PORT} =="
"${BIN}" --host "${PEER}" --port "${PORT}" --iters 400

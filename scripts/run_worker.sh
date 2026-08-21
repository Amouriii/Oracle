#!/usr/bin/env bash
# Start an Oracle worker.  The node id decides which layer shard it owns.
#
#   NODE_ID=1 MODEL=/models/m.gguf scripts/run_worker.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/oracle-engine-worker"
CONFIG="${CONFIG:-${ROOT}/configs/cluster.toml}"

if [[ -z "${NODE_ID:-}" ]]; then
  echo "NODE_ID is required (which node in ${CONFIG} is this machine?)" >&2
  exit 2
fi
if [[ ! -x "${BIN}" ]]; then
  echo "not built yet; running scripts/build.sh" >&2
  "${ROOT}/scripts/build.sh"
fi
if [[ -z "${ORACLE_CLUSTER_SECRET:-}" ]]; then
  echo "warning: ORACLE_CLUSTER_SECRET is not set; the master's handshake will be refused" >&2
fi

args=(--config "${CONFIG}" --id "${NODE_ID}")
[[ -n "${MODEL:-}" ]] && args+=(--model "${MODEL}")
[[ -n "${RUNNER:-}" ]] && args+=(--runner "${RUNNER}")
[[ -n "${THREADS:-}" ]] && args+=(--threads "${THREADS}")

exec "${BIN}" "${args[@]}"

#!/usr/bin/env bash
# Start the Oracle master.
#
#   scripts/run_master.sh                          # configs/cluster.toml
#   MODEL=/models/m.gguf scripts/run_master.sh      # override the model path
#   CONFIG=configs/single.toml scripts/run_master.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/oracle-engine-master"
CONFIG="${CONFIG:-${ROOT}/configs/cluster.toml}"
NODE_ID="${NODE_ID:-0}"

if [[ ! -x "${BIN}" ]]; then
  echo "not built yet; running scripts/build.sh" >&2
  "${ROOT}/scripts/build.sh"
fi

args=(--config "${CONFIG}" --id "${NODE_ID}")
[[ -n "${MODEL:-}" ]] && args+=(--model "${MODEL}")
[[ -n "${RUNNER:-}" ]] && args+=(--runner "${RUNNER}")
[[ -n "${PORT:-}" ]] && args+=(--port "${PORT}")
[[ -n "${THREADS:-}" ]] && args+=(--threads "${THREADS}")
[[ "${SINGLE:-0}" == "1" ]] && args+=(--single)
[[ "${NO_AUTH:-0}" == "1" ]] && args+=(--no-auth)

if [[ "${NO_AUTH:-0}" != "1" && -z "${ORACLE_API_KEYS:-}" && ! -f "${ROOT}/configs/api_keys" ]]; then
  echo "warning: no API keys configured." >&2
  echo "  generate one:  ${BIN} --generate-key" >&2
  echo "  then export:   ORACLE_API_KEYS=demo:<secret>" >&2
  echo "  or run with:   NO_AUTH=1 $0" >&2
fi

exec "${BIN}" "${args[@]}"

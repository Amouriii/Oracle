#!/usr/bin/env bash
set -euo pipefail
# Phase 4: OpenAI-compatible HTTP on :8000. Start with a small GGUF, then 70B after phase 2.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:-${ROOT}/configs/cluster.toml}"
PORT="${2:-8000}"
BIN="${ROOT}/build/oracle-engine-master"
if [[ ! -x "${BIN}" ]]; then
  cmake -S "${ROOT}" -B "${ROOT}/build" -DORACLE_BUILD_PYTHON=OFF
  cmake --build "${ROOT}/build" --target oracle-engine-master
fi
echo "Starting single-node master (workers: oracle-engine-worker --config ${CFG} --id 1|2)"
"${BIN}" --config "${CFG}" --single --runner accelerate &
PID=$!
trap 'kill ${PID} 2>/dev/null || true' EXIT
sleep 1
echo "== GET /v1/models =="
curl -sS "http://127.0.0.1:${PORT}/v1/models" || true
echo
echo "== POST /v1/chat/completions =="
curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Oracle","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' || true
echo

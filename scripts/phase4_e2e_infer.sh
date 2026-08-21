#!/usr/bin/env bash
# Phase 4 - end to end: a real two-node mesh over TCP, the OpenAI API, the
# security gate, concurrency, worker death and recovery.
#
# With no model of your own this uses a synthetic one; point MODEL at a .gguf
# to exercise the same paths with real weights on a single node instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -z "${MODEL:-}" ]]; then
  exec "${ROOT}/scripts/e2e_test.sh"
fi

BIN="${ROOT}/build/oracle-engine-master"
PORT="${PORT:-8000}"
[[ -x "${BIN}" ]] || "${ROOT}/scripts/build.sh"

"${BIN}" --config "${ROOT}/configs/single.toml" --single --no-auth \
  --model "${MODEL}" --port "${PORT}" &
PID=$!
trap 'kill ${PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  sleep 0.5
  curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/health" && break
done

echo "== GET /v1/models =="
curl -sS "http://127.0.0.1:${PORT}/v1/models"; echo
echo
echo "== POST /v1/chat/completions =="
curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three primary colours."}],"max_tokens":48}'
echo

#!/usr/bin/env bash
# End-to-end check: build a synthetic GGUF, run it across a real two-node mesh
# over TCP, and exercise the API, the security gate and failure handling.
#
# Everything runs on localhost, so this needs no model download and no second
# machine; the code paths are the same ones a Thunderbolt mesh uses.
set -euo pipefail
# Byte-wise matching: the assertions below grep JSON that carries model output,
# and in a UTF-8 locale BSD grep stops matching a line at the first byte that is
# not valid UTF-8 -- so a pattern after such a byte silently never matches.
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
WORK="$(mktemp -d)"
BASE_PORT="${BASE_PORT:-18400}"
HTTP_PORT=$((BASE_PORT))
KEY="sk-oracle-e2e-$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
PIDS=()

# Preserve the failing status explicitly: without this the trap's own last
# command decides the exit code, and a failed check reports success.
cleanup() {
  local status=$?
  for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  sleep 0.3
  for pid in "${PIDS[@]:-}"; do kill -9 "${pid}" 2>/dev/null || true; done
  rm -rf "${WORK}"
  exit "${status}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  echo "--- cluster ---" >&2
  curl -sS -H "Authorization: Bearer ${KEY}" --max-time 3 \
    "http://127.0.0.1:${HTTP_PORT}/cluster" >&2 || true
  echo >&2
  for f in "${WORK}"/*.log; do
    [[ -f "${f}" ]] || continue
    echo "--- ${f} ---" >&2
    tail -30 "${f}" >&2
  done
  exit 1
}
ok()   { echo "  ok  $*"; }

[[ -x "${BUILD}/oracle-engine-master" ]] || fail "build first: scripts/build.sh"
[[ -x "${BUILD}/test_gguf" ]] || fail "tests were not built; run scripts/build.sh"

echo "== building a synthetic model =="
# test_gguf writes and then removes its own fixture, so build a tiny writer that
# leaves one on disk for the mesh to load.
cat > "${WORK}/mkmodel.cpp" <<'CPP'
#include "gguf_fixture.hpp"
#include <cstdio>
int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: mkmodel OUT.gguf\n"); return 2; }
  const auto m = oracle_test::build_tiny_gguf(argv[1]);
  if (!m.n_vocab) { std::fprintf(stderr, "could not write %s\n", argv[1]); return 1; }
  std::printf("%s: %u layers, vocab %u\n", argv[1], m.n_layers, m.n_vocab);
  return 0;
}
CPP
c++ -std=c++20 -O1 -I"${ROOT}/tests" "${WORK}/mkmodel.cpp" -o "${WORK}/mkmodel"
"${WORK}/mkmodel" "${WORK}/tiny.gguf"

"${BUILD}/oracle-model-info" "${WORK}/tiny.gguf" | sed 's/^/  /'

cat > "${WORK}/mesh.toml" <<TOML
[cluster]
name = "oracle-e2e"
http_port = ${HTTP_PORT}
heartbeat_interval_ms = 150
heartbeat_misses = 4
max_sequences = 4

[model]
path = "${WORK}/tiny.gguf"
max_seq = 128
act_dtype = "f16"

[server]
max_concurrent = 2
max_queue_depth = 8
queue_timeout_ms = 10000

[security]
require_api_key = true
require_worker_auth = true
cluster_secret = "e2e-cluster-secret"
requests_per_minute = 6000
burst = 200
max_concurrent_requests = 8
max_concurrent_per_key = 8
audit_log = "${WORK}/security.log"

[[nodes]]
id = 0
role = "master"
host = "127.0.0.1"
transport_port = $((BASE_PORT + 1))
heartbeat_port = $((BASE_PORT + 3))
ram_budget_gb = 8

[[nodes]]
id = 1
role = "worker"
host = "127.0.0.1"
transport_port = $((BASE_PORT + 2))
heartbeat_port = $((BASE_PORT + 4))
ram_budget_gb = 8
TOML

echo
echo "== starting a two-node mesh =="
"${BUILD}/oracle-engine-worker" --config "${WORK}/mesh.toml" --id 1 > "${WORK}/worker.log" 2>&1 &
PIDS+=($!)
sleep 1
"${BUILD}/oracle-engine-master" --config "${WORK}/mesh.toml" --id 0 --api-key "${KEY}" \
  > "${WORK}/master.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 80); do
  sleep 0.25
  curl -fsS -o /dev/null "http://127.0.0.1:${HTTP_PORT}/health" && break
done
curl -fsS -o /dev/null "http://127.0.0.1:${HTTP_PORT}/health" \
  || { cat "${WORK}/master.log" "${WORK}/worker.log"; fail "master never became healthy"; }

api() { curl -sS -H "Authorization: Bearer ${KEY}" "$@"; }
code() { curl -sS -o /dev/null -w '%{http_code}' "$@"; }

echo
echo "== endpoints =="
api "http://127.0.0.1:${HTTP_PORT}/health" | grep -q '"status":"ok"' \
  || fail "/health did not report ok"; ok "/health"
api "http://127.0.0.1:${HTTP_PORT}/v1/models" | grep -q '"object":"model"' \
  || fail "/v1/models"; ok "/v1/models"
api "http://127.0.0.1:${HTTP_PORT}/metrics" | grep -q 'oracle_requests_total' \
  || fail "/metrics"; ok "/metrics"
curl -fsS "http://127.0.0.1:${HTTP_PORT}/" | grep -q '<title>Oracle</title>' \
  || fail "dashboard"; ok "dashboard at /"

CLUSTER="$(api "http://127.0.0.1:${HTTP_PORT}/cluster")"
echo "${CLUSTER}" | grep -q '"nodes":2' || fail "cluster does not report 2 nodes"
echo "${CLUSTER}" | grep -q '"lm_head":true' || fail "no lm_head stage in the pipeline"
ok "/cluster reports a 2-stage pipeline"

echo
echo "== security =="
[[ "$(code -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
      -d '{"messages":[{"role":"user","content":"x"}]}')" == "401" ]] \
  || fail "an unauthenticated request was not rejected"; ok "401 without a key"
[[ "$(code -X POST -H 'Authorization: Bearer nope-nope-nope' \
      "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
      -d '{"messages":[{"role":"user","content":"x"}]}')" == "401" ]] \
  || fail "a bad key was accepted"; ok "401 with a bad key"
api -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" -d '{oops' \
  | grep -q 'not valid JSON' || fail "malformed JSON was not reported"; ok "malformed JSON rejected"
api -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"max_tokens":999999}' \
  | grep -q 'exceeds the limit' || fail "max_tokens was not clamped"; ok "oversized max_tokens rejected"

echo
echo "== inference across the mesh =="
OUT="$(api -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello world"}],"max_tokens":8,"temperature":0}')"
echo "${OUT}" | grep -q '"object":"chat.completion"' || { echo "${OUT}"; fail "chat completion"; }
echo "${OUT}" | grep -q '"completion_tokens":8' || { echo "${OUT}"; fail "wrong token count"; }
# Whatever the model produced, the response has to be parseable JSON: a client
# that stops at the first bad byte would otherwise lose the response silently.
if command -v python3 >/dev/null 2>&1; then
  printf '%s' "${OUT}" | python3 -c "
import json, sys
d = json.loads(sys.stdin.buffer.read().decode('utf-8'))
assert d['choices'][0]['message']['role'] == 'assistant'
assert d['usage']['completion_tokens'] == 8
" || fail "the completion was not valid, parseable JSON"
  ok "response is valid UTF-8 JSON"
fi
ok "buffered chat completion (8 tokens through 2 stages)"

STREAM="$(api -N -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}')"
echo "${STREAM}" | grep -q 'chat.completion.chunk' || fail "no SSE chunks"
echo "${STREAM}" | grep -q 'data: \[DONE\]' || fail "stream did not terminate"
ok "streaming chat completion"

api -X POST "http://127.0.0.1:${HTTP_PORT}/v1/completions" \
  -d '{"prompt":"hello","max_tokens":4}' | grep -q '"object":"text_completion"' \
  || fail "/v1/completions"; ok "/v1/completions"

echo
echo "== concurrency =="
# Wait only on the curls: a bare `wait` would also block on the node processes,
# which do not exit until the script does.
curl_pids=()
for i in 1 2 3 4; do
  api -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"c'"${i}"'"}],"max_tokens":6}' \
    > "${WORK}/c${i}.json" &
  curl_pids+=($!)
done
for pid in "${curl_pids[@]}"; do wait "${pid}" || fail "a concurrent request failed"; done
for i in 1 2 3 4; do
  grep -q '"object":"chat.completion"' "${WORK}/c${i}.json" \
    || { cat "${WORK}/c${i}.json"; fail "concurrent request ${i}"; }
done
ok "4 concurrent requests all completed"

echo
echo "== failure handling =="
kill -9 "${PIDS[0]}" || fail "the worker was not running when we tried to kill it"
for _ in $(seq 1 40); do
  sleep 0.25
  api "http://127.0.0.1:${HTTP_PORT}/health" | grep -q '"status":"degraded"' && break
done
api "http://127.0.0.1:${HTTP_PORT}/health" | grep -q '"status":"degraded"' \
  || fail "a dead worker did not degrade the cluster"; ok "dead worker detected"

REFUSED="$(api -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"max_tokens":4}')"
echo "${REFUSED}" | grep -q 'no healthy worker' \
  || { echo "${REFUSED}"; fail "a request was dispatched to a dead stage"; }
ok "requests refused with a clear reason while degraded"

"${BUILD}/oracle-engine-worker" --config "${WORK}/mesh.toml" --id 1 \
  > "${WORK}/worker2.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 120); do
  sleep 0.25
  api "http://127.0.0.1:${HTTP_PORT}/health" | grep -q '"status":"ok"' && break
done
api "http://127.0.0.1:${HTTP_PORT}/health" | grep -q '"status":"ok"' \
  || { cat "${WORK}/worker2.log"; fail "the worker did not rejoin"; }
ok "worker rejoined and the cluster recovered"

api -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"back"}],"max_tokens":4}' \
  | grep -q '"object":"chat.completion"' || fail "inference after recovery"
ok "inference works again after recovery"

echo
echo "== audit trail =="
grep -q '"category":"worker"' "${WORK}/security.log" || fail "no worker events were logged"
ok "security events recorded in the audit log"

echo
echo "ALL END-TO-END CHECKS PASSED"
exit 0

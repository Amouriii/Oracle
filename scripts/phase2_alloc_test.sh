#!/usr/bin/env bash
set -euo pipefail
# Phase 2: MemoryShardManager dry-run for 70B-Q4 vs RAM budgets. No weights required.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/test_shard_plan"
CFG="${ROOT}/configs/cluster.toml"
echo "== shard plan (${CFG}) =="
if [[ -x "${BIN}" ]]; then
  "${BIN}"
else
  echo "building tests..."
  cmake -S "${ROOT}" -B "${ROOT}/build" -DORACLE_BUILD_PYTHON=OFF
  cmake --build "${ROOT}/build" --target test_shard_plan
  "${ROOT}/build/test_shard_plan"
fi
echo "PressureMonitor refuses overcommit against ram_budget_gb in cluster.toml."
echo "70B Q4 (~35-40 GiB weights + KV) must fit pooled node RAM, not 2-4 GiB dGPU."

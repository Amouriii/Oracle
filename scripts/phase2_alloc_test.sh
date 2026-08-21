#!/usr/bin/env bash
# Phase 2 - allocation: does the model fit across the configured RAM budgets?
#
#   scripts/phase2_alloc_test.sh [model.gguf] [gb,gb,gb]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
SPLIT="${2:-32,32,32}"

if [[ ! -x "${ROOT}/build/oracle-model-info" ]]; then
  "${ROOT}/scripts/build.sh"
fi

echo "== shard planner against configs/cluster.toml =="
ctest --test-dir "${ROOT}/build" -R shard_plan --output-on-failure

if [[ -n "${MODEL}" ]]; then
  echo
  echo "== ${MODEL} across budgets ${SPLIT} GiB =="
  "${ROOT}/build/oracle-model-info" "${MODEL}" --split "${SPLIT}"
else
  echo
  echo "pass a .gguf path to get a real per-node plan:"
  echo "  $0 /models/model.gguf 32,32,32"
fi

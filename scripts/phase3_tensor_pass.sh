#!/usr/bin/env bash
# Phase 3 - tensors: wire format, shared-memory ring, TCP round trip, the
# dequantisers, and a model split across two nodes producing identical logits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "${ROOT}" -B "${ROOT}/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${ROOT}/build" -j"$( (command -v nproc >/dev/null && nproc) || echo 4)"
ctest --test-dir "${ROOT}/build" \
  -R "protocol|ring_buffer|transport_tcp|quant|gguf|tokenizer|pipeline_tiny" \
  --output-on-failure

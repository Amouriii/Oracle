#!/usr/bin/env bash
set -euo pipefail
# Phase 3: tiny F16/F32 buffers + Accelerate/Metal GEMM + checksum.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "${ROOT}" -B "${ROOT}/build" -DORACLE_BUILD_PYTHON=OFF
cmake --build "${ROOT}/build" --target test_pipeline_tiny test_protocol test_transport_tcp
ctest --test-dir "${ROOT}/build" -R "pipeline_tiny|protocol|transport_tcp" --output-on-failure
echo "Tensor pass: packed ORCL header, writev TCP, identity GEMM checksum at master."

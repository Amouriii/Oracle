#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
curl -fsSL -o "${ROOT}/third_party/httplib.h" \
  https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h
echo "wrote third_party/httplib.h"

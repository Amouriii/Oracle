#!/usr/bin/env bash
# Configure, build and test Oracle.  Works on Linux and macOS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4)"

if [[ ! -f "${ROOT}/third_party/httplib.h" ]]; then
  "${ROOT}/scripts/fetch_deps.sh"
fi

cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "${BUILD}" -j"${JOBS}"
ctest --test-dir "${BUILD}" --output-on-failure

echo
echo "binaries in ${BUILD}:"
for b in oracle-engine-master oracle-engine-worker oracle-engine-bench-net oracle-model-info; do
  [[ -x "${BUILD}/${b}" ]] && echo "  ${BUILD}/${b}"
done

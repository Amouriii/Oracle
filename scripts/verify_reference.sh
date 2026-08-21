#!/usr/bin/env bash
# Cross-check Oracle's transformer against an independent NumPy implementation.
#
# tests/test_gguf.cpp compares the engine against itself (one node vs two); this
# compares it against a second implementation written straight from the
# architecture, which is what catches a sign error in RoPE, a wrong grouped-query
# head mapping or a transposed projection.
#
#   scripts/verify_reference.sh                 # synthetic fixture
#   scripts/verify_reference.sh model.gguf      # an F32/F16 model of your own
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
TOKENS="${TOKENS:-1,260,261,262}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

PY=python3
if ! "${PY}" -c "import numpy" 2>/dev/null; then
  # Many system pythons are externally managed (PEP 668) and refuse a plain
  # `pip install`, so fall back to a throwaway virtualenv rather than telling
  # the user to fight their package manager.
  echo "numpy not found; creating a temporary virtualenv"
  if "${PY}" -m venv "${WORK}/venv" 2>/dev/null &&
     "${WORK}/venv/bin/python" -m pip install --quiet --disable-pip-version-check numpy; then
    PY="${WORK}/venv/bin/python"
  else
    echo "could not obtain numpy; install it and re-run (pip install numpy)" >&2
    exit 2
  fi
fi
[[ -x "${BUILD}/oracle-model-info" ]] || "${ROOT}/scripts/build.sh"

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  cat > "${WORK}/mkmodel.cpp" <<'CPP'
#include "gguf_fixture.hpp"
#include <cstdio>
int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: mkmodel OUT.gguf\n"); return 2; }
  return oracle_test::build_tiny_gguf(argv[1]).n_vocab ? 0 : 1;
}
CPP
  c++ -std=c++20 -O1 -I"${ROOT}/tests" "${WORK}/mkmodel.cpp" -o "${WORK}/mkmodel"
  MODEL="${WORK}/model.gguf"
  "${WORK}/mkmodel" "${MODEL}"
  echo "using the synthetic fixture at ${MODEL}"
fi

echo "tokens: ${TOKENS}"
"${BUILD}/oracle-model-info" "${MODEL}" --logits "${TOKENS}" > "${WORK}/engine.txt"
"${PY}" "${ROOT}/tests/reference_llama.py" "${MODEL}" \
  --tokens "${TOKENS}" --compare "${WORK}/engine.txt"

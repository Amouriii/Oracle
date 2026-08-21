#!/usr/bin/env bash
# Download a small GGUF to try Oracle with, and report what Oracle sees in it.
#
#   scripts/fetch_model.sh                 # ~1.1 GB TinyLlama Q4_K_M
#   scripts/fetch_model.sh <url> <dest>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
URL="${1:-${DEFAULT_URL}}"
DEST="${2:-${ROOT}/models/$(basename "${URL%%\?*}")}"

mkdir -p "$(dirname "${DEST}")"
if [[ -f "${DEST}" ]]; then
  echo "already have ${DEST}"
else
  echo "downloading ${URL}"
  curl -fL --progress-bar -o "${DEST}.part" "${URL}"
  mv "${DEST}.part" "${DEST}"
fi

echo
if [[ -x "${ROOT}/build/oracle-model-info" ]]; then
  "${ROOT}/build/oracle-model-info" "${DEST}"
  echo
  echo "record its digest so tampering is detected:"
  echo "  sha256sum '${DEST}' | awk '{print \$1\"  \"\$2}' >> configs/models.sha256"
else
  echo "build first (scripts/build.sh) to inspect it with oracle-model-info"
fi
echo
echo "run it:  MODEL='${DEST}' SINGLE=1 NO_AUTH=1 scripts/run_master.sh"

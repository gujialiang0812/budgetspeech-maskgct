#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-fs/BudgetSpeechDatasets}"
PRESET="${PRESET:-librispeech_warmup}"
EXTRACT="${EXTRACT:-1}"

ARCHIVE_DIR="${ROOT}/archives"
EXTRACT_DIR="${ROOT}/extracted"
mkdir -p "${ARCHIVE_DIR}" "${EXTRACT_DIR}"

download_one() {
  local id="$1"
  local url="$2"
  local archive="$3"
  local extract_subdir="$4"
  local archive_path="${ARCHIVE_DIR}/${archive}"

  echo "[download] ${id}"
  echo "  url: ${url}"
  echo "  out: ${archive_path}"
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "${archive_path}" "${url}"

  if [[ "${EXTRACT}" == "1" ]]; then
    local dest="${EXTRACT_DIR}/${extract_subdir}"
    mkdir -p "${dest}"
    echo "[extract] ${archive_path} -> ${dest}"
    case "${archive}" in
      *.tar.gz|*.tgz)
        tar -xzf "${archive_path}" -C "${dest}"
        ;;
      *.zip)
        unzip -o "${archive_path}" -d "${dest}"
        ;;
      *)
        echo "unknown archive type: ${archive}" >&2
        return 1
        ;;
    esac
  fi
}

case "${PRESET}" in
  librispeech_warmup)
    download_one "libritts_train_clean_100" \
      "https://www.openslr.org/resources/60/train-clean-100.tar.gz" \
      "libritts_train-clean-100.tar.gz" \
      "LibriTTS"
    ;;
  libritts_eval)
    download_one "libritts_dev_clean" \
      "https://www.openslr.org/resources/60/dev-clean.tar.gz" \
      "libritts_dev-clean.tar.gz" \
      "LibriTTS"
    download_one "libritts_test_clean" \
      "https://www.openslr.org/resources/60/test-clean.tar.gz" \
      "libritts_test-clean.tar.gz" \
      "LibriTTS"
    ;;
  aishell3)
    download_one "aishell3" \
      "https://www.openslr.org/resources/93/data_aishell3.tgz" \
      "data_aishell3.tgz" \
      "AISHELL-3"
    ;;
  paper_core)
    PRESET=librispeech_warmup EXTRACT="${EXTRACT}" ROOT="${ROOT}" "$0"
    PRESET=libritts_eval EXTRACT="${EXTRACT}" ROOT="${ROOT}" "$0"
    PRESET=aishell3 EXTRACT="${EXTRACT}" ROOT="${ROOT}" "$0"
    ;;
  *)
    echo "Unknown PRESET=${PRESET}" >&2
    echo "Available: librispeech_warmup, libritts_eval, aishell3, paper_core" >&2
    exit 1
    ;;
esac

echo "[done] dataset preset ${PRESET} under ${ROOT}"

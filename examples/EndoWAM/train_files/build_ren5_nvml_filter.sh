#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_INCLUDE="${CUDA_INCLUDE:-/usr/local/cuda-12.4/targets/x86_64-linux/include}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data-hdd2/ljs/.cache/dit4dit/nvml_filter}"
REAL_NVML_LIB="${REAL_NVML_LIB:-$(readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1)}"
CC="${CC:-gcc}"

if [[ ! -f "${CUDA_INCLUDE}/nvml.h" ]]; then
  echo "NVML header was not found: ${CUDA_INCLUDE}/nvml.h" >&2
  exit 1
fi
if [[ ! -f "${REAL_NVML_LIB}" ]]; then
  echo "Real NVML library was not found: ${REAL_NVML_LIB}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
"${CC}" \
  -shared -fPIC -O2 -Wall -Wextra -Werror \
  -I"${CUDA_INCLUDE}" \
  -DREAL_NVML_PATH="\"${REAL_NVML_LIB}\"" \
  -Wl,-soname,libnvidia-ml.so.1 \
  "${SCRIPT_DIR}/ren5_nvml_filter.c" \
  -ldl -pthread \
  -o "${OUTPUT_DIR}/libnvidia-ml.so.1"

echo "ren5 process-local NVML filter built: ${OUTPUT_DIR}/libnvidia-ml.so.1"
